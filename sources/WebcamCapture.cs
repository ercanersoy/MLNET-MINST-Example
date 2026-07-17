// WebcamCapture.cs - Webcam capture using the Video for Windows (avicap32)
//                    API, which ships with Windows and requires no external
//                    library.
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

namespace ObjectDetector
{
    // Captures frames from a video capture device and raises an event for each
    // decoded frame. The caller owns the delivered bitmap and must dispose it.
    public class WebcamCapture : IDisposable
    {
        // avicap32 message identifiers (relative to WM_USER = 0x400).
        private const int WM_USER = 0x0400;
        private const int WM_CAP_SET_CALLBACK_FRAME = WM_USER + 5;
        private const int WM_CAP_DRIVER_CONNECT = WM_USER + 10;
        private const int WM_CAP_DRIVER_DISCONNECT = WM_USER + 11;
        private const int WM_CAP_GET_VIDEOFORMAT = WM_USER + 44;
        private const int WM_CAP_SET_VIDEOFORMAT = WM_USER + 45;
        private const int WM_CAP_SET_PREVIEW = WM_USER + 50;
        private const int WM_CAP_SET_PREVIEWRATE = WM_USER + 52;

        // BITMAPINFOHEADER.biCompression values. BI_RGB is uncompressed RGB;
        // the rest are FourCC codes packed little-endian into an int. Consumer
        // webcams overwhelmingly stream one of these packed-YUV formats or MJPG
        // rather than raw RGB, so we must recognise and decode them - treating
        // their bytes as RGB is what produces "random coloured noise".
        private const int BI_RGB = 0;
        private const int FOURCC_YUY2 = 0x32595559; // 'Y','U','Y','2'
        private const int FOURCC_YUYV = 0x56595559; // 'Y','U','Y','V'
        private const int FOURCC_UYVY = 0x59565955; // 'U','Y','V','Y'
        private const int FOURCC_MJPG = 0x47504A4D; // 'M','J','P','G'
        private const int FOURCC_JPEG = 0x4745504A; // 'J','P','E','G'

        // Live preview frame rate. The preview timer drives the frame callback
        // at this rate; recordings should use the same value so that they play
        // back at the correct speed.
        public const int TargetFps = 30;
        private const int PreviewRateMs = 1000 / TargetFps;

        // Standard child window style flags.
        private const int WS_CHILD = 0x40000000;
        private const int WS_VISIBLE = 0x10000000;

        [DllImport("avicap32.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr capCreateCaptureWindowA(
            string lpszWindowName, int dwStyle, int x, int y, int nWidth, int nHeight,
            IntPtr hwndParent, int nID);

        [DllImport("user32.dll")]
        private static extern int SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll")]
        private static extern bool DestroyWindow(IntPtr hWnd);

        // Callback invoked by avicap32 for each streamed frame. The VfW API
        // expects a __stdcall (CALLBACK) function pointer.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate int CaptureCallback(IntPtr hWnd, IntPtr lpVHdr);

        [StructLayout(LayoutKind.Sequential)]
        private struct VIDEOHDR
        {
            public IntPtr lpData;
            public int dwBufferLength;
            public int dwBytesUsed;
            public int dwTimeCaptured;
            public IntPtr dwUser;
            public int dwFlags;
            public int dwReserved0;
            public int dwReserved1;
            public int dwReserved2;
            public int dwReserved3;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct BITMAPINFOHEADER
        {
            public int biSize;
            public int biWidth;
            public int biHeight;
            public short biPlanes;
            public short biBitCount;
            public int biCompression;
            public int biSizeImage;
            public int biXPelsPerMeter;
            public int biYPelsPerMeter;
            public int biClrUsed;
            public int biClrImportant;
        }

        private IntPtr _capHwnd = IntPtr.Zero;
        private CaptureCallback _callback;   // kept alive to prevent GC
        private int _frameWidth;
        private int _frameHeight;
        private int _bitCount;
        private int _compression;
        private bool _topDown;               // true if scanlines are stored top-to-bottom
        private bool _running;

        // Raised for every previewed frame on the UI thread (via the preview
        // timer). Handlers should do as little work as possible so the user
        // interface stays responsive.
        public event EventHandler<FrameEventArgs> FrameReady;

        public bool IsRunning
        {
            get { return _running; }
        }

        // Starts streaming from the given device index. The capture window is
        // created as a hidden child of the supplied parent handle.
        public bool Start(IntPtr parentHandle, int deviceIndex)
        {
            Stop();

            // The capture window MUST be visible: avicap32 invokes the frame
            // callback only just before it *displays* a preview frame, so a
            // hidden (non-WS_VISIBLE) or fully occluded window never receives
            // any frames at all. We therefore create a real, visible child
            // window but keep it tiny and tucked into the top-left corner so it
            // is imperceptible - the callback still delivers full-resolution
            // frames, since the frame size comes from the video format, not the
            // window size.
            _capHwnd = capCreateCaptureWindowA("ObjectDetectorCapture", WS_CHILD | WS_VISIBLE, 0, 0, 2, 2, parentHandle, 0);
            if (_capHwnd == IntPtr.Zero)
            {
                return false;
            }

            int connected = SendMessage(_capHwnd, WM_CAP_DRIVER_CONNECT, (IntPtr)deviceIndex, IntPtr.Zero);
            if (connected == 0)
            {
                DestroyWindow(_capHwnd);
                _capHwnd = IntPtr.Zero;
                return false;
            }

            if (!ConfigureFormat() || !CanDecode())
            {
                Stop();
                return false;
            }

            // Install a per-frame callback and start a live preview. Preview
            // mode delivers each frame to the callback through the window's
            // message loop, so - unlike a streaming capture sequence - it never
            // blocks the calling thread or freezes the user interface.
            _callback = new CaptureCallback(OnFrame);
            IntPtr callbackPtr = Marshal.GetFunctionPointerForDelegate(_callback);
            SendMessage(_capHwnd, WM_CAP_SET_CALLBACK_FRAME, IntPtr.Zero, callbackPtr);
            SendMessage(_capHwnd, WM_CAP_SET_PREVIEWRATE, (IntPtr)PreviewRateMs, IntPtr.Zero);

            // Enabling preview makes the driver stream frames to our callback,
            // which we render ourselves. The tiny capture window shows the raw
            // preview, but at 2x2 pixels it is invisible in practice.
            int previewing = SendMessage(_capHwnd, WM_CAP_SET_PREVIEW, (IntPtr)1, IntPtr.Zero);
            if (previewing == 0)
            {
                Stop();
                return false;
            }

            _running = true;
            return true;
        }

        public void Stop()
        {
            if (_capHwnd != IntPtr.Zero)
            {
                SendMessage(_capHwnd, WM_CAP_SET_PREVIEW, IntPtr.Zero, IntPtr.Zero);
                SendMessage(_capHwnd, WM_CAP_SET_CALLBACK_FRAME, IntPtr.Zero, IntPtr.Zero);
                SendMessage(_capHwnd, WM_CAP_DRIVER_DISCONNECT, IntPtr.Zero, IntPtr.Zero);
                DestroyWindow(_capHwnd);
                _capHwnd = IntPtr.Zero;
            }
            _callback = null;
            _running = false;
        }

        // Reads the device's current video format and, when it is not already
        // uncompressed RGB, asks the driver to switch to 24-bit RGB. Most VfW /
        // UVC drivers honour this request, which lets us take the simple RGB
        // path; when a driver refuses, we fall back to decoding whatever packed
        // format it insists on streaming (see BuildBitmap).
        private bool ConfigureFormat()
        {
            int size = SendMessage(_capHwnd, WM_CAP_GET_VIDEOFORMAT, IntPtr.Zero, IntPtr.Zero);
            if (size < Marshal.SizeOf(typeof(BITMAPINFOHEADER)))
            {
                return false;
            }

            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                SendMessage(_capHwnd, WM_CAP_GET_VIDEOFORMAT, (IntPtr)size, buffer);
                BITMAPINFOHEADER header = (BITMAPINFOHEADER)Marshal.PtrToStructure(buffer, typeof(BITMAPINFOHEADER));

                // Try to force plain RGB24 at the device's current resolution.
                // Changing only the pixel format (not the dimensions) is the
                // request drivers are most likely to accept.
                if (header.biCompression != BI_RGB || (header.biBitCount != 24 && header.biBitCount != 32))
                {
                    TrySetRgb24(header.biWidth, Math.Abs(header.biHeight));

                    // Re-read: use whatever the driver actually settled on,
                    // whether or not the request above succeeded.
                    SendMessage(_capHwnd, WM_CAP_GET_VIDEOFORMAT, (IntPtr)size, buffer);
                    header = (BITMAPINFOHEADER)Marshal.PtrToStructure(buffer, typeof(BITMAPINFOHEADER));
                }

                _frameWidth = header.biWidth;
                _frameHeight = Math.Abs(header.biHeight);
                _bitCount = header.biBitCount;
                _compression = header.biCompression;

                // For BI_RGB, a positive biHeight means the DIB is bottom-up;
                // negative means top-down. Packed-YUV and MJPG frames are always
                // delivered top-down.
                _topDown = (_compression != BI_RGB) || (header.biHeight < 0);

                return _frameWidth > 0 && _frameHeight > 0;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        private void TrySetRgb24(int width, int height)
        {
            BITMAPINFOHEADER rgb = new BITMAPINFOHEADER();
            rgb.biSize = Marshal.SizeOf(typeof(BITMAPINFOHEADER));
            rgb.biWidth = width;
            rgb.biHeight = height;          // positive: request a bottom-up RGB DIB
            rgb.biPlanes = 1;
            rgb.biBitCount = 24;
            rgb.biCompression = BI_RGB;
            rgb.biSizeImage = 0;

            IntPtr ptr = Marshal.AllocHGlobal(rgb.biSize);
            try
            {
                Marshal.StructureToPtr(rgb, ptr, false);
                SendMessage(_capHwnd, WM_CAP_SET_VIDEOFORMAT, (IntPtr)rgb.biSize, ptr);
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        // Whether the negotiated format is one BuildBitmap knows how to turn
        // into a bitmap.
        private bool CanDecode()
        {
            switch (_compression)
            {
                case BI_RGB:
                    return _bitCount == 24 || _bitCount == 32;
                case FOURCC_YUY2:
                case FOURCC_YUYV:
                case FOURCC_UYVY:
                case FOURCC_MJPG:
                case FOURCC_JPEG:
                    return true;
                default:
                    return false;
            }
        }

        // Converts the raw device frame into a managed, upright bitmap and
        // raises the FrameReady event. Invoked by the preview timer on the
        // thread that owns the capture window (the UI thread).
        private int OnFrame(IntPtr hWnd, IntPtr lpVHdr)
        {
            EventHandler<FrameEventArgs> handler = FrameReady;
            if (handler == null || lpVHdr == IntPtr.Zero)
            {
                return 0;
            }

            try
            {
                VIDEOHDR vhdr = (VIDEOHDR)Marshal.PtrToStructure(lpVHdr, typeof(VIDEOHDR));

                // Some drivers report the frame size in dwBufferLength and
                // leave dwBytesUsed at zero; treat either as a valid frame so
                // we don't silently drop every frame.
                int frameBytes = (vhdr.dwBytesUsed > 0) ? vhdr.dwBytesUsed : vhdr.dwBufferLength;
                if (vhdr.lpData == IntPtr.Zero || frameBytes <= 0)
                {
                    return 0;
                }

                Bitmap bitmap = BuildBitmap(vhdr.lpData, frameBytes);
                if (bitmap != null)
                {
                    handler(this, new FrameEventArgs(bitmap));
                }
            }
            catch (Exception)
            {
                // Never let a decode error propagate into native code.
            }

            return 0;
        }

        // Turns one raw device frame into a self-contained upright RGB bitmap.
        // The source memory is only valid for the duration of the callback, so
        // every path returns a bitmap that owns its own pixels.
        private Bitmap BuildBitmap(IntPtr data, int length)
        {
            switch (_compression)
            {
                case BI_RGB:
                    return BuildRgbBitmap(data);
                case FOURCC_YUY2:
                case FOURCC_YUYV:
                    return BuildYuyvBitmap(data, false);
                case FOURCC_UYVY:
                    return BuildYuyvBitmap(data, true);
                case FOURCC_MJPG:
                case FOURCC_JPEG:
                    return BuildJpegBitmap(data, length);
                default:
                    return null;
            }
        }

        private Bitmap BuildRgbBitmap(IntPtr data)
        {
            PixelFormat format = (_bitCount == 32) ? PixelFormat.Format32bppRgb : PixelFormat.Format24bppRgb;
            int bytesPerPixel = _bitCount / 8;
            int stride = ((_frameWidth * bytesPerPixel) + 3) & ~3;

            IntPtr scan0;
            int scanStride;
            if (_topDown)
            {
                scan0 = data;
                scanStride = stride;
            }
            else
            {
                // Bottom-up DIB: point at the last scanline and walk upward with
                // a negative stride to obtain an upright image.
                scan0 = new IntPtr(data.ToInt64() + ((long)stride * (_frameHeight - 1)));
                scanStride = -stride;
            }

            using (Bitmap reference = new Bitmap(_frameWidth, _frameHeight, scanStride, format, scan0))
            {
                // Clone into a self-contained bitmap; the source memory is only
                // valid for the duration of the callback.
                return new Bitmap(reference);
            }
        }

        // Decodes a packed 4:2:2 YUV frame (YUY2/YUYV, or UYVY when 'swapped')
        // into a 24bpp RGB bitmap. Each four source bytes carry two pixels that
        // share one U and one V sample.
        private Bitmap BuildYuyvBitmap(IntPtr data, bool swapped)
        {
            int width = _frameWidth;
            int height = _frameHeight;
            int srcStride = ((width * 2) + 3) & ~3;

            int srcLen = srcStride * height;
            byte[] src = new byte[srcLen];
            Marshal.Copy(data, src, 0, srcLen);

            Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            BitmapData dst = bitmap.LockBits(
                new Rectangle(0, 0, width, height),
                ImageLockMode.WriteOnly,
                PixelFormat.Format24bppRgb);
            try
            {
                int dstStride = dst.Stride;
                byte[] row = new byte[dstStride];
                // Offsets of the two luma and the two chroma bytes within each
                // 4-byte group, depending on the packing order.
                int y0i = swapped ? 1 : 0;
                int u0i = swapped ? 0 : 1;
                int y1i = swapped ? 3 : 2;
                int v0i = swapped ? 2 : 3;

                for (int y = 0; y < height; y++)
                {
                    int srcRow = y * srcStride;
                    int di = 0;
                    for (int x = 0; x + 1 < width; x += 2)
                    {
                        int si = srcRow + (x * 2);
                        int y0 = src[si + y0i];
                        int u = src[si + u0i];
                        int y1 = src[si + y1i];
                        int v = src[si + v0i];

                        WriteBgr(row, di, y0, u, v);
                        WriteBgr(row, di + 3, y1, u, v);
                        di += 6;
                    }
                    Marshal.Copy(row, 0, new IntPtr(dst.Scan0.ToInt64() + ((long)y * dstStride)), dstStride);
                }
            }
            finally
            {
                bitmap.UnlockBits(dst);
            }
            return bitmap;
        }

        // Writes one BT.601 YUV sample as a B, G, R triple (24bpp DIB order).
        private static void WriteBgr(byte[] dst, int offset, int y, int u, int v)
        {
            int c = y - 16;
            int d = u - 128;
            int e = v - 128;

            int r = (298 * c + 409 * e + 128) >> 8;
            int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b = (298 * c + 516 * d + 128) >> 8;

            dst[offset + 0] = Clip(b);
            dst[offset + 1] = Clip(g);
            dst[offset + 2] = Clip(r);
        }

        private static byte Clip(int value)
        {
            if (value < 0)
            {
                return 0;
            }
            if (value > 255)
            {
                return 255;
            }
            return (byte)value;
        }

        // Decodes an MJPG/JPEG frame. GDI+ reads the compressed bytes directly.
        private Bitmap BuildJpegBitmap(IntPtr data, int length)
        {
            byte[] bytes = new byte[length];
            Marshal.Copy(data, bytes, 0, length);
            using (MemoryStream stream = new MemoryStream(bytes, false))
            using (Image decoded = Image.FromStream(stream))
            {
                return new Bitmap(decoded);
            }
        }

        public void Dispose()
        {
            Stop();
        }
    }

    // Carries a captured frame to event subscribers.
    public class FrameEventArgs : EventArgs
    {
        public Bitmap Frame;

        public FrameEventArgs(Bitmap frame)
        {
            Frame = frame;
        }
    }
}
