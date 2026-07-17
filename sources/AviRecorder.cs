// AviRecorder.cs - Native Motion-JPEG AVI writer for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.
//
// Writes a standard RIFF/AVI file with Motion-JPEG compressed frames. JPEG
// encoding uses the built-in System.Drawing encoder, so no external library is
// required. AVI is the only container that can be produced natively; the other
// formats requested by the user would require an external codec.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;

namespace ObjectDetector
{
    public class AviRecorder : IDisposable
    {
        private FileStream _stream;
        private BinaryWriter _writer;
        private readonly object _lock = new object();

        private int _width;
        private int _height;
        private int _fps;

        private long _riffSizePos;
        private long _totalFramesPos;
        private long _streamLengthPos;
        private long _moviSizePos;
        private long _moviFourccPos;

        private readonly List<uint> _indexOffsets = new List<uint>();
        private readonly List<uint> _indexSizes = new List<uint>();

        private int _maxChunkSize;
        private bool _open;

        public bool IsRecording
        {
            get { return _open; }
        }

        public string FilePath { get; private set; }

        public void Start(string path, int width, int height, int fps)
        {
            lock (_lock)
            {
                if (_open)
                {
                    throw new InvalidOperationException("Recording is already in progress.");
                }

                _width = width;
                _height = height;
                _fps = (fps > 0) ? fps : 15;
                _indexOffsets.Clear();
                _indexSizes.Clear();
                _maxChunkSize = 0;
                FilePath = path;

                _stream = new FileStream(path, FileMode.Create, FileAccess.Write);
                _writer = new BinaryWriter(_stream);
                WriteHeaders();
                _open = true;
            }
        }

        // Encodes a bitmap to JPEG and appends it as a frame.
        public void AddFrame(Bitmap frame)
        {
            lock (_lock)
            {
                if (!_open)
                {
                    return;
                }

                byte[] jpeg;
                using (MemoryStream ms = new MemoryStream())
                {
                    // Scale to the recorder resolution if the source differs.
                    if (frame.Width != _width || frame.Height != _height)
                    {
                        using (Bitmap scaled = new Bitmap(_width, _height, PixelFormat.Format24bppRgb))
                        {
                            using (Graphics g = Graphics.FromImage(scaled))
                            {
                                g.DrawImage(frame, 0, 0, _width, _height);
                            }
                            scaled.Save(ms, ImageFormat.Jpeg);
                        }
                    }
                    else
                    {
                        frame.Save(ms, ImageFormat.Jpeg);
                    }
                    jpeg = ms.ToArray();
                }

                uint offset = (uint)(_stream.Position - _moviFourccPos);
                _indexOffsets.Add(offset);
                _indexSizes.Add((uint)jpeg.Length);
                if (jpeg.Length > _maxChunkSize)
                {
                    _maxChunkSize = jpeg.Length;
                }

                WriteFourCc("00dc");
                _writer.Write((uint)jpeg.Length);
                _writer.Write(jpeg);
                if ((jpeg.Length & 1) == 1)
                {
                    _writer.Write((byte)0); // chunks are word aligned
                }
            }
        }

        // Finalises the file: writes the index and patches the placeholder
        // size and count fields.
        public void Stop()
        {
            lock (_lock)
            {
                if (!_open)
                {
                    return;
                }

                long moviEnd = _stream.Position;
                uint moviSize = (uint)(moviEnd - _moviFourccPos);

                // idx1 chunk.
                WriteFourCc("idx1");
                _writer.Write((uint)(_indexOffsets.Count * 16));
                for (int i = 0; i < _indexOffsets.Count; i++)
                {
                    WriteFourCc("00dc");
                    _writer.Write((uint)0x10); // AVIIF_KEYFRAME
                    _writer.Write(_indexOffsets[i]);
                    _writer.Write(_indexSizes[i]);
                }

                long fileEnd = _stream.Position;
                int frameCount = _indexOffsets.Count;

                Patch(_riffSizePos, (uint)(fileEnd - 8));
                Patch(_moviSizePos, moviSize);
                Patch(_totalFramesPos, (uint)frameCount);
                Patch(_streamLengthPos, (uint)frameCount);

                _stream.Seek(fileEnd, SeekOrigin.Begin);
                _writer.Flush();
                _writer.Close();
                _stream = null;
                _writer = null;
                _open = false;
            }
        }

        private void WriteHeaders()
        {
            uint microSecPerFrame = (uint)(1000000 / _fps);

            WriteFourCc("RIFF");
            _riffSizePos = _stream.Position;
            _writer.Write((uint)0); // patched on Stop
            WriteFourCc("AVI ");

            // LIST 'hdrl' (fixed size).
            WriteFourCc("LIST");
            _writer.Write((uint)192);
            WriteFourCc("hdrl");

            // 'avih' main header (56 bytes).
            WriteFourCc("avih");
            _writer.Write((uint)56);
            _writer.Write(microSecPerFrame);
            _writer.Write((uint)0);       // dwMaxBytesPerSec
            _writer.Write((uint)0);       // dwPaddingGranularity
            _writer.Write((uint)0x10);    // dwFlags = AVIF_HASINDEX
            _totalFramesPos = _stream.Position;
            _writer.Write((uint)0);       // dwTotalFrames (patched)
            _writer.Write((uint)0);       // dwInitialFrames
            _writer.Write((uint)1);       // dwStreams
            _writer.Write((uint)0);       // dwSuggestedBufferSize
            _writer.Write((uint)_width);
            _writer.Write((uint)_height);
            _writer.Write((uint)0);       // dwReserved[0]
            _writer.Write((uint)0);       // dwReserved[1]
            _writer.Write((uint)0);       // dwReserved[2]
            _writer.Write((uint)0);       // dwReserved[3]

            // LIST 'strl'.
            WriteFourCc("LIST");
            _writer.Write((uint)124);
            WriteFourCc("strl");

            // 'strh' stream header (56 bytes).
            WriteFourCc("strh");
            _writer.Write((uint)56);
            WriteFourCc("vids");
            WriteFourCc("MJPG");
            _writer.Write((uint)0);       // dwFlags
            _writer.Write((short)0);      // wPriority
            _writer.Write((short)0);      // wLanguage
            _writer.Write((uint)0);       // dwInitialFrames
            _writer.Write((uint)1);       // dwScale
            _writer.Write((uint)_fps);    // dwRate
            _writer.Write((uint)0);       // dwStart
            _streamLengthPos = _stream.Position;
            _writer.Write((uint)0);       // dwLength (patched)
            _writer.Write((uint)0);       // dwSuggestedBufferSize
            _writer.Write((uint)0xFFFFFFFF); // dwQuality
            _writer.Write((uint)0);       // dwSampleSize
            _writer.Write((short)0);      // rcFrame.left
            _writer.Write((short)0);      // rcFrame.top
            _writer.Write((short)_width); // rcFrame.right
            _writer.Write((short)_height);// rcFrame.bottom

            // 'strf' bitmap info header (40 bytes).
            WriteFourCc("strf");
            _writer.Write((uint)40);
            _writer.Write((uint)40);      // biSize
            _writer.Write(_width);        // biWidth
            _writer.Write(_height);       // biHeight
            _writer.Write((short)1);      // biPlanes
            _writer.Write((short)24);     // biBitCount
            WriteFourCc("MJPG");          // biCompression
            _writer.Write((uint)(_width * _height * 3)); // biSizeImage
            _writer.Write((uint)0);       // biXPelsPerMeter
            _writer.Write((uint)0);       // biYPelsPerMeter
            _writer.Write((uint)0);       // biClrUsed
            _writer.Write((uint)0);       // biClrImportant

            // LIST 'movi'.
            WriteFourCc("LIST");
            _moviSizePos = _stream.Position;
            _writer.Write((uint)0);       // patched on Stop
            _moviFourccPos = _stream.Position;
            WriteFourCc("movi");
        }

        private void Patch(long position, uint value)
        {
            _stream.Seek(position, SeekOrigin.Begin);
            _writer.Write(value);
        }

        private void WriteFourCc(string fourCc)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(fourCc);
            _writer.Write(bytes, 0, 4);
        }

        public void Dispose()
        {
            try
            {
                if (_open)
                {
                    Stop();
                }
            }
            catch (Exception)
            {
            }
        }
    }
}
