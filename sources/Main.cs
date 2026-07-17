// main.cs - Application entry point and main window for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.
//
// A WinForms object detection (image classification) application that runs an
// Ultralytics YOLO26 ONNX model on a live webcam feed. It shows the detected
// object, keeps a history list, and can save snapshots and record video.

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Windows.Forms;

namespace ObjectDetector
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }

    public class MainForm : Form
    {
        private AppSettings _settings;
        private OnnxClassifier _classifier;
        private readonly WebcamCapture _capture = new WebcamCapture();
        private readonly AviRecorder _recorder = new AviRecorder();

        // UI controls.
        private PictureBox _preview;
        private Label _detectedLabel;
        private Label _recordingLabel;
        private ListBox _history;
        private Button _cameraButton;
        private Button _snapshotButton;
        private Button _recordButton;
        private Button _settingsButton;
        private Button _aboutButton;
        private Button _clearHistoryButton;
        private Label _statusLabel;

        // Classification worker state.
        private readonly object _classifyLock = new object();
        private Bitmap _classifyFrame;
        private Thread _classifyThread;
        private volatile bool _classifyRunning;
        private string _lastHistoryLabel = string.Empty;

        private Size _frameSize = Size.Empty;
        private volatile bool _closing;

        public MainForm()
        {
            _settings = AppSettings.Load();
            Localization.SetLanguage(_settings.Language);

            BuildUi();
            ApplyTexts();

            Load += OnFormLoad;
            FormClosing += OnFormClosing;
        }

        // ---- UI construction -------------------------------------------------

        private void BuildUi()
        {
            ClientSize = new Size(900, 566);
            MinimumSize = new Size(760, 520);
            StartPosition = FormStartPosition.CenterScreen;

            _preview = new PictureBox();
            _preview.BackColor = Color.Black;
            _preview.SizeMode = PictureBoxSizeMode.Zoom;
            _preview.Location = new Point(12, 12);
            _preview.Size = new Size(560, 420);
            _preview.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom;

            _recordingLabel = new Label();
            _recordingLabel.ForeColor = Color.Red;
            _recordingLabel.BackColor = Color.Black;
            _recordingLabel.AutoSize = true;
            _recordingLabel.Font = new Font(Font.FontFamily, 10f, FontStyle.Bold);
            _recordingLabel.Location = new Point(20, 20);
            _recordingLabel.Visible = false;

            _detectedLabel = new Label();
            _detectedLabel.Font = new Font(Font.FontFamily, 13f, FontStyle.Bold);
            _detectedLabel.AutoEllipsis = true;
            _detectedLabel.Location = new Point(12, 442);
            _detectedLabel.Size = new Size(560, 34);
            _detectedLabel.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom;

            Label historyTitle = new Label();
            historyTitle.Name = "historyTitle";
            historyTitle.AutoSize = true;
            historyTitle.Location = new Point(588, 12);
            historyTitle.Anchor = AnchorStyles.Top | AnchorStyles.Right;

            _history = new ListBox();
            _history.Location = new Point(588, 36);
            _history.Size = new Size(300, 396);
            _history.Anchor = AnchorStyles.Top | AnchorStyles.Right | AnchorStyles.Bottom;
            _history.HorizontalScrollbar = true;

            _clearHistoryButton = new Button();
            _clearHistoryButton.Location = new Point(588, 442);
            _clearHistoryButton.Size = new Size(300, 30);
            _clearHistoryButton.Anchor = AnchorStyles.Right | AnchorStyles.Bottom;
            _clearHistoryButton.Click += delegate { _history.Items.Clear(); };

            AnchorStyles leftBottom = AnchorStyles.Left | AnchorStyles.Bottom;
            AnchorStyles rightBottom = AnchorStyles.Right | AnchorStyles.Bottom;

            // Camera controls sit under the preview (left aligned); the settings
            // and about buttons sit under the history list (right aligned) so
            // that each group stays aligned with its column when the window is
            // resized.
            _cameraButton = MakeButton(12, 486, 120, leftBottom, OnToggleCamera);
            _snapshotButton = MakeButton(140, 486, 120, leftBottom, OnSnapshot);
            _recordButton = MakeButton(268, 486, 120, leftBottom, OnToggleRecording);
            _settingsButton = MakeButton(588, 486, 146, rightBottom, OnSettings);
            _aboutButton = MakeButton(742, 486, 146, rightBottom, OnAbout);

            _statusLabel = new Label();
            _statusLabel.AutoSize = false;
            _statusLabel.BorderStyle = BorderStyle.Fixed3D;
            _statusLabel.TextAlign = ContentAlignment.MiddleLeft;
            _statusLabel.Location = new Point(12, 528);
            _statusLabel.Size = new Size(876, 26);
            _statusLabel.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom;

            _snapshotButton.Enabled = false;
            _recordButton.Enabled = false;

            Controls.Add(_preview);
            _preview.Controls.Add(_recordingLabel);
            Controls.Add(_detectedLabel);
            Controls.Add(historyTitle);
            Controls.Add(_history);
            Controls.Add(_clearHistoryButton);
            Controls.Add(_cameraButton);
            Controls.Add(_snapshotButton);
            Controls.Add(_recordButton);
            Controls.Add(_settingsButton);
            Controls.Add(_aboutButton);
            Controls.Add(_statusLabel);
        }

        private Button MakeButton(int x, int y, int width, AnchorStyles anchor, EventHandler handler)
        {
            Button button = new Button();
            button.Location = new Point(x, y);
            button.Size = new Size(width, 32);
            button.Anchor = anchor;
            button.Click += handler;
            return button;
        }

        // Applies translated captions to every control. Called on start-up and
        // whenever the language changes.
        private void ApplyTexts()
        {
            Text = Localization.Get("App.Title");
            _cameraButton.Text = _capture.IsRunning
                ? Localization.Get("Main.StopCamera")
                : Localization.Get("Main.StartCamera");
            _snapshotButton.Text = Localization.Get("Main.Snapshot");
            _recordButton.Text = _recorder.IsRecording
                ? Localization.Get("Main.StopRecord")
                : Localization.Get("Main.StartRecord");
            _settingsButton.Text = Localization.Get("Main.Settings");
            _aboutButton.Text = Localization.Get("Main.About");
            _clearHistoryButton.Text = Localization.Get("Main.ClearHistory");
            _recordingLabel.Text = Localization.Get("Main.Recording");
            _detectedLabel.Text = Localization.Get("Main.Detected") + " " + Localization.Get("Main.None");

            Control[] found = Controls.Find("historyTitle", false);
            if (found.Length > 0)
            {
                found[0].Text = Localization.Get("Main.History");
            }
        }

        // ---- Life cycle ------------------------------------------------------

        private void OnFormLoad(object sender, EventArgs e)
        {
            LoadModel();
        }

        private void LoadModel()
        {
            try
            {
                string modelPath = FindModel();
                if (modelPath == null)
                {
                    string expected = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "yolo26x-cls.onnx");
                    ShowError(Localization.Format("Error.ModelNotFound", expected));
                    return;
                }

                _classifier = new OnnxClassifier(modelPath);
                SetStatus(Localization.Format("Status.ModelLoaded", Path.GetFileName(modelPath)));
            }
            catch (Exception ex)
            {
                ShowError(Localization.Format("Error.ModelLoad", ex.Message));
            }
        }

        // Locates the first ONNX model next to the executable.
        private static string FindModel()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string preferred = Path.Combine(baseDir, "yolo26x-cls.onnx");
            if (File.Exists(preferred))
            {
                return preferred;
            }

            string[] candidates = Directory.GetFiles(baseDir, "*.onnx");
            return (candidates.Length > 0) ? candidates[0] : null;
        }

        private void OnFormClosing(object sender, FormClosingEventArgs e)
        {
            _closing = true;
            StopClassifyThread();

            if (_recorder.IsRecording)
            {
                _recorder.Stop();
            }
            _recorder.Dispose();

            _capture.FrameReady -= OnFrameReady;
            _capture.Dispose();

            if (_classifier != null)
            {
                _classifier.Dispose();
            }

            lock (_classifyLock)
            {
                if (_classifyFrame != null)
                {
                    _classifyFrame.Dispose();
                    _classifyFrame = null;
                }
            }
        }

        // ---- Camera ----------------------------------------------------------

        private void OnToggleCamera(object sender, EventArgs e)
        {
            if (_capture.IsRunning)
            {
                StopCamera();
            }
            else
            {
                StartCamera();
            }
        }

        private void StartCamera()
        {
            _capture.FrameReady -= OnFrameReady;
            _capture.FrameReady += OnFrameReady;

            if (!_capture.Start(Handle, _settings.CameraIndex))
            {
                _capture.FrameReady -= OnFrameReady;
                ShowError(Localization.Get("Error.CameraStart"));
                return;
            }

            StartClassifyThread();

            _cameraButton.Text = Localization.Get("Main.StopCamera");
            _snapshotButton.Enabled = true;
            _recordButton.Enabled = true;
            SetStatus(Localization.Get("Status.CameraStarted"));
        }

        private void StopCamera()
        {
            if (_recorder.IsRecording)
            {
                StopRecording();
            }

            _capture.FrameReady -= OnFrameReady;
            _capture.Stop();
            StopClassifyThread();

            _cameraButton.Text = Localization.Get("Main.StartCamera");
            _snapshotButton.Enabled = false;
            _recordButton.Enabled = false;
            SetStatus(Localization.Get("Status.CameraStopped"));
        }

        // Runs on the capture thread for each incoming frame.
        private void OnFrameReady(object sender, FrameEventArgs e)
        {
            if (_closing)
            {
                e.Frame.Dispose();
                return;
            }

            Bitmap processed = null;
            try
            {
                processed = ApplyImageSettings(e.Frame);
            }
            finally
            {
                e.Frame.Dispose();
            }

            if (processed == null)
            {
                return;
            }

            _frameSize = processed.Size;

            // Feed the recorder (encodes synchronously and retains nothing).
            if (_recorder.IsRecording)
            {
                try
                {
                    _recorder.AddFrame(processed);
                }
                catch (Exception)
                {
                }
            }

            // Hand a private copy to the classification worker.
            lock (_classifyLock)
            {
                if (_classifyFrame != null)
                {
                    _classifyFrame.Dispose();
                }
                _classifyFrame = (Bitmap)processed.Clone();
            }

            // Display on the UI thread; it takes ownership of 'processed'.
            try
            {
                BeginInvoke((MethodInvoker)delegate { ShowFrame(processed); });
            }
            catch (Exception)
            {
                processed.Dispose();
            }
        }

        private void ShowFrame(Bitmap frame)
        {
            if (_closing)
            {
                frame.Dispose();
                return;
            }

            Image previous = _preview.Image;
            _preview.Image = frame;
            if (previous != null)
            {
                previous.Dispose();
            }
        }

        // Applies mirror, brightness and contrast to a captured frame.
        private Bitmap ApplyImageSettings(Bitmap source)
        {
            int w = source.Width;
            int h = source.Height;
            Bitmap result = new Bitmap(w, h, PixelFormat.Format24bppRgb);

            float contrast = 1f + (_settings.Contrast / 100f);
            float brightness = _settings.Brightness / 100f;
            float translate = 0.5f * (1f - contrast);
            float offset = translate + brightness;

            ColorMatrix matrix = new ColorMatrix(new float[][]
            {
                new float[] { contrast, 0, 0, 0, 0 },
                new float[] { 0, contrast, 0, 0, 0 },
                new float[] { 0, 0, contrast, 0, 0 },
                new float[] { 0, 0, 0, 1, 0 },
                new float[] { offset, offset, offset, 0, 1 }
            });

            using (Graphics g = Graphics.FromImage(result))
            using (ImageAttributes attributes = new ImageAttributes())
            {
                attributes.SetColorMatrix(matrix);

                // Destination parallelogram: upper-left, upper-right, lower-left.
                Point[] destination = _settings.MirrorHorizontal
                    ? new Point[] { new Point(w, 0), new Point(0, 0), new Point(w, h) }
                    : new Point[] { new Point(0, 0), new Point(w, 0), new Point(0, h) };

                g.DrawImage(source, destination, new Rectangle(0, 0, w, h), GraphicsUnit.Pixel, attributes);
            }

            return result;
        }

        // ---- Classification worker ------------------------------------------

        private void StartClassifyThread()
        {
            if (_classifier == null || _classifyRunning)
            {
                return;
            }

            _classifyRunning = true;
            _classifyThread = new Thread(ClassifyLoop);
            _classifyThread.IsBackground = true;
            _classifyThread.Start();
        }

        private void StopClassifyThread()
        {
            _classifyRunning = false;
            Thread thread = _classifyThread;
            if (thread != null && thread.IsAlive)
            {
                thread.Join(2000);
            }
            _classifyThread = null;
        }

        private void ClassifyLoop()
        {
            while (_classifyRunning)
            {
                int interval = _settings.ClassifyIntervalMs;
                Thread.Sleep(interval > 0 ? interval : 500);

                if (!_classifyRunning)
                {
                    break;
                }

                Bitmap frame = null;
                lock (_classifyLock)
                {
                    if (_classifyFrame != null)
                    {
                        frame = _classifyFrame;
                        _classifyFrame = null;
                    }
                }

                if (frame == null || _classifier == null)
                {
                    continue;
                }

                try
                {
                    System.Collections.Generic.List<Prediction> predictions = _classifier.Classify(frame, 5);
                    if (predictions.Count > 0)
                    {
                        Prediction top = predictions[0];
                        if (top.Confidence * 100f >= _settings.ConfidenceThreshold)
                        {
                            PublishPrediction(top);
                        }
                    }
                }
                catch (Exception)
                {
                }
                finally
                {
                    frame.Dispose();
                }
            }
        }

        private void PublishPrediction(Prediction prediction)
        {
            if (_closing)
            {
                return;
            }

            try
            {
                BeginInvoke((MethodInvoker)delegate
                {
                    if (_closing)
                    {
                        return;
                    }

                    int percent = (int)Math.Round(prediction.Confidence * 100f);
                    string confidence = Localization.Format("Confidence", percent);
                    _detectedLabel.Text = Localization.Get("Main.Detected") + " " + prediction.Label + " - " + confidence;

                    if (!string.Equals(prediction.Label, _lastHistoryLabel, StringComparison.Ordinal))
                    {
                        _lastHistoryLabel = prediction.Label;
                        string entry = string.Format(CultureInfo.CurrentCulture, "{0:HH:mm:ss}  {1}  ({2})",
                            DateTime.Now, prediction.Label, confidence);
                        _history.Items.Insert(0, entry);

                        // Keep the history list from growing without bound.
                        while (_history.Items.Count > 500)
                        {
                            _history.Items.RemoveAt(_history.Items.Count - 1);
                        }
                    }
                });
            }
            catch (Exception)
            {
            }
        }

        // ---- Snapshot --------------------------------------------------------

        private void OnSnapshot(object sender, EventArgs e)
        {
            Image current = _preview.Image;
            if (current == null)
            {
                ShowError(Localization.Get("Error.NoFrame"));
                return;
            }

            try
            {
                EnsureOutputDirectory();

                string extension;
                ImageFormat format = AppSettings.GetImageFormat(_settings.SnapshotImageFormat, out extension);
                string fileName = "snapshot_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) + extension;
                string path = Path.Combine(_settings.OutputDirectory, fileName);

                using (Bitmap copy = new Bitmap(current))
                {
                    copy.Save(path, format);
                }

                SetStatus(Localization.Format("Status.SnapshotSaved", path));
            }
            catch (Exception ex)
            {
                ShowError(Localization.Format("Error.SnapshotSave", ex.Message));
            }
        }

        // ---- Recording -------------------------------------------------------

        private void OnToggleRecording(object sender, EventArgs e)
        {
            if (_recorder.IsRecording)
            {
                StopRecording();
            }
            else
            {
                StartRecording();
            }
        }

        private void StartRecording()
        {
            // Only AVI can be produced without an external library.
            if (_settings.RecordingFormat != VideoFormat.Avi)
            {
                string name = _settings.RecordingFormat.ToString().ToUpperInvariant();
                ShowError(Localization.Format("Error.VideoFormatUnsupported", name));
                return;
            }

            if (_frameSize == Size.Empty)
            {
                ShowError(Localization.Get("Error.NoFrame"));
                return;
            }

            try
            {
                EnsureOutputDirectory();

                string extension = AppSettings.GetVideoExtension(_settings.RecordingFormat);
                string fileName = "recording_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) + extension;
                string path = Path.Combine(_settings.OutputDirectory, fileName);

                _recorder.Start(path, _frameSize.Width, _frameSize.Height, WebcamCapture.TargetFps);

                _recordButton.Text = Localization.Get("Main.StopRecord");
                _recordingLabel.Visible = true;
                SetStatus(Localization.Format("Status.RecordingStarted", path));
            }
            catch (Exception ex)
            {
                ShowError(Localization.Format("Error.RecordStart", ex.Message));
            }
        }

        private void StopRecording()
        {
            string path = _recorder.FilePath;
            _recorder.Stop();

            _recordButton.Text = Localization.Get("Main.StartRecord");
            _recordingLabel.Visible = false;
            SetStatus(Localization.Format("Status.RecordingStopped", path));
        }

        // ---- Dialogs ---------------------------------------------------------

        private void OnSettings(object sender, EventArgs e)
        {
            using (SettingsForm dialog = new SettingsForm(_settings))
            {
                if (dialog.ShowDialog(this) == DialogResult.OK)
                {
                    AppLanguage previousLanguage = _settings.Language;
                    int previousCamera = _settings.CameraIndex;

                    _settings = dialog.Result;
                    _settings.Save();

                    if (_settings.Language != previousLanguage)
                    {
                        Localization.SetLanguage(_settings.Language);
                        ApplyTexts();
                    }

                    // Restart the camera if the device changed while running.
                    if (_settings.CameraIndex != previousCamera && _capture.IsRunning)
                    {
                        StopCamera();
                        StartCamera();
                    }
                }
            }
        }

        private void OnAbout(object sender, EventArgs e)
        {
            using (AboutForm dialog = new AboutForm())
            {
                dialog.ShowDialog(this);
            }
        }

        // ---- Helpers ---------------------------------------------------------

        private void EnsureOutputDirectory()
        {
            if (string.IsNullOrEmpty(_settings.OutputDirectory))
            {
                _settings.OutputDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
            }
            if (!Directory.Exists(_settings.OutputDirectory))
            {
                Directory.CreateDirectory(_settings.OutputDirectory);
            }
        }

        private void SetStatus(string text)
        {
            _statusLabel.Text = text;
        }

        private void ShowError(string message)
        {
            MessageBox.Show(this, message, Localization.Get("Error.Title"),
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
