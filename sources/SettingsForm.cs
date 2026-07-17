// SettingsForm.cs - Settings dialog for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.

using System;
using System.Drawing;
using System.Windows.Forms;

namespace ObjectDetector
{
    public class SettingsForm : Form
    {
        private readonly AppSettings _settings;

        private ComboBox _languageCombo;
        private NumericUpDown _cameraIndex;
        private CheckBox _mirror;
        private TrackBar _brightness;
        private TrackBar _contrast;
        private NumericUpDown _confidence;
        private NumericUpDown _interval;
        private ComboBox _snapshotFormat;
        private ComboBox _videoFormat;
        private TextBox _outputDir;

        // Returns the edited settings after the dialog is confirmed.
        public AppSettings Result
        {
            get { return _settings; }
        }

        public SettingsForm(AppSettings current)
        {
            _settings = current.Clone();

            Text = Localization.Get("Settings.Title");
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition = FormStartPosition.CenterParent;
            MaximizeBox = false;
            MinimizeBox = false;
            ClientSize = new Size(440, 520);

            BuildLanguageGroup();
            BuildImageGroup();
            BuildDetectionGroup();
            BuildOutputGroup();
            BuildButtons();

            LoadValues();
        }

        private void BuildLanguageGroup()
        {
            GroupBox group = new GroupBox();
            group.Text = Localization.Get("Settings.LanguageGroup");
            group.Location = new Point(12, 12);
            group.Size = new Size(416, 60);

            Label label = new Label();
            label.Text = Localization.Get("Settings.Language");
            label.AutoSize = true;
            label.Location = new Point(12, 26);

            _languageCombo = new ComboBox();
            _languageCombo.DropDownStyle = ComboBoxStyle.DropDownList;
            _languageCombo.Location = new Point(160, 22);
            _languageCombo.Size = new Size(240, 24);
            _languageCombo.Items.Add(Localization.Get("Lang.System"));
            _languageCombo.Items.Add(Localization.Get("Lang.Turkish"));
            _languageCombo.Items.Add(Localization.Get("Lang.EnglishUK"));
            _languageCombo.Items.Add(Localization.Get("Lang.EnglishUS"));

            group.Controls.Add(label);
            group.Controls.Add(_languageCombo);
            Controls.Add(group);
        }

        private void BuildImageGroup()
        {
            GroupBox group = new GroupBox();
            group.Text = Localization.Get("Settings.ImageGroup");
            group.Location = new Point(12, 80);
            group.Size = new Size(416, 180);

            Label cameraLabel = new Label();
            cameraLabel.Text = Localization.Get("Settings.Camera");
            cameraLabel.AutoSize = true;
            cameraLabel.Location = new Point(12, 28);

            _cameraIndex = new NumericUpDown();
            _cameraIndex.Minimum = 0;
            _cameraIndex.Maximum = 9;
            _cameraIndex.Location = new Point(260, 24);
            _cameraIndex.Size = new Size(140, 24);

            _mirror = new CheckBox();
            _mirror.Text = Localization.Get("Settings.Mirror");
            _mirror.AutoSize = true;
            _mirror.Location = new Point(12, 58);

            Label brightnessLabel = new Label();
            brightnessLabel.Text = Localization.Get("Settings.Brightness");
            brightnessLabel.AutoSize = true;
            brightnessLabel.Location = new Point(12, 92);

            _brightness = new TrackBar();
            _brightness.Minimum = -100;
            _brightness.Maximum = 100;
            _brightness.TickFrequency = 25;
            _brightness.Location = new Point(150, 86);
            _brightness.Size = new Size(250, 40);

            Label contrastLabel = new Label();
            contrastLabel.Text = Localization.Get("Settings.Contrast");
            contrastLabel.AutoSize = true;
            contrastLabel.Location = new Point(12, 136);

            _contrast = new TrackBar();
            _contrast.Minimum = -100;
            _contrast.Maximum = 100;
            _contrast.TickFrequency = 25;
            _contrast.Location = new Point(150, 130);
            _contrast.Size = new Size(250, 40);

            group.Controls.Add(cameraLabel);
            group.Controls.Add(_cameraIndex);
            group.Controls.Add(_mirror);
            group.Controls.Add(brightnessLabel);
            group.Controls.Add(_brightness);
            group.Controls.Add(contrastLabel);
            group.Controls.Add(_contrast);
            Controls.Add(group);
        }

        private void BuildDetectionGroup()
        {
            GroupBox group = new GroupBox();
            group.Text = Localization.Get("Settings.DetectionGroup");
            group.Location = new Point(12, 268);
            group.Size = new Size(416, 90);

            Label confidenceLabel = new Label();
            confidenceLabel.Text = Localization.Get("Settings.Confidence");
            confidenceLabel.AutoSize = true;
            confidenceLabel.Location = new Point(12, 26);

            _confidence = new NumericUpDown();
            _confidence.Minimum = 0;
            _confidence.Maximum = 100;
            _confidence.Location = new Point(260, 22);
            _confidence.Size = new Size(140, 24);

            Label intervalLabel = new Label();
            intervalLabel.Text = Localization.Get("Settings.Interval");
            intervalLabel.AutoSize = true;
            intervalLabel.Location = new Point(12, 56);

            _interval = new NumericUpDown();
            _interval.Minimum = 100;
            _interval.Maximum = 5000;
            _interval.Increment = 100;
            _interval.Location = new Point(260, 52);
            _interval.Size = new Size(140, 24);

            group.Controls.Add(confidenceLabel);
            group.Controls.Add(_confidence);
            group.Controls.Add(intervalLabel);
            group.Controls.Add(_interval);
            Controls.Add(group);
        }

        private void BuildOutputGroup()
        {
            GroupBox group = new GroupBox();
            group.Text = Localization.Get("Settings.OutputGroup");
            group.Location = new Point(12, 366);
            group.Size = new Size(416, 110);

            Label snapshotLabel = new Label();
            snapshotLabel.Text = Localization.Get("Settings.SnapshotFormat");
            snapshotLabel.AutoSize = true;
            snapshotLabel.Location = new Point(12, 26);

            _snapshotFormat = new ComboBox();
            _snapshotFormat.DropDownStyle = ComboBoxStyle.DropDownList;
            _snapshotFormat.Location = new Point(200, 22);
            _snapshotFormat.Size = new Size(200, 24);
            _snapshotFormat.Items.AddRange(new object[] { "BMP", "JPEG", "TIFF", "PNG" });

            Label videoLabel = new Label();
            videoLabel.Text = Localization.Get("Settings.VideoFormat");
            videoLabel.AutoSize = true;
            videoLabel.Location = new Point(12, 54);

            _videoFormat = new ComboBox();
            _videoFormat.DropDownStyle = ComboBoxStyle.DropDownList;
            _videoFormat.Location = new Point(200, 50);
            _videoFormat.Size = new Size(200, 24);
            _videoFormat.Items.AddRange(new object[]
            {
                "AVI", "WMV", "MOV", "MKV", "MPEG", "MPEG-2", "MPEG-4"
            });

            Label dirLabel = new Label();
            dirLabel.Text = Localization.Get("Settings.OutputDir");
            dirLabel.AutoSize = true;
            dirLabel.Location = new Point(12, 82);

            _outputDir = new TextBox();
            _outputDir.Location = new Point(120, 79);
            _outputDir.Size = new Size(200, 24);
            _outputDir.ReadOnly = true;

            Button browse = new Button();
            browse.Text = Localization.Get("Settings.Browse");
            browse.Location = new Point(326, 78);
            browse.Size = new Size(74, 26);
            browse.Click += OnBrowse;

            group.Controls.Add(snapshotLabel);
            group.Controls.Add(_snapshotFormat);
            group.Controls.Add(videoLabel);
            group.Controls.Add(_videoFormat);
            group.Controls.Add(dirLabel);
            group.Controls.Add(_outputDir);
            group.Controls.Add(browse);
            Controls.Add(group);
        }

        private void BuildButtons()
        {
            Button ok = new Button();
            ok.Text = Localization.Get("Settings.Ok");
            ok.Size = new Size(90, 30);
            ok.Location = new Point(238, 484);
            ok.Click += OnOk;

            Button cancel = new Button();
            cancel.Text = Localization.Get("Settings.Cancel");
            cancel.DialogResult = DialogResult.Cancel;
            cancel.Size = new Size(90, 30);
            cancel.Location = new Point(338, 484);

            Controls.Add(ok);
            Controls.Add(cancel);
            AcceptButton = ok;
            CancelButton = cancel;
        }

        private void LoadValues()
        {
            _languageCombo.SelectedIndex = (int)_settings.Language;
            _cameraIndex.Value = Clamp(_settings.CameraIndex, 0, 9);
            _mirror.Checked = _settings.MirrorHorizontal;
            _brightness.Value = Clamp(_settings.Brightness, -100, 100);
            _contrast.Value = Clamp(_settings.Contrast, -100, 100);
            _confidence.Value = Clamp(_settings.ConfidenceThreshold, 0, 100);
            _interval.Value = Clamp(_settings.ClassifyIntervalMs, 100, 5000);
            _snapshotFormat.SelectedIndex = (int)_settings.SnapshotImageFormat;
            _videoFormat.SelectedIndex = (int)_settings.RecordingFormat;
            _outputDir.Text = _settings.OutputDirectory;
        }

        private void OnBrowse(object sender, EventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                dialog.SelectedPath = _outputDir.Text;
                if (dialog.ShowDialog(this) == DialogResult.OK)
                {
                    _outputDir.Text = dialog.SelectedPath;
                }
            }
        }

        private void OnOk(object sender, EventArgs e)
        {
            _settings.Language = (AppLanguage)_languageCombo.SelectedIndex;
            _settings.CameraIndex = (int)_cameraIndex.Value;
            _settings.MirrorHorizontal = _mirror.Checked;
            _settings.Brightness = _brightness.Value;
            _settings.Contrast = _contrast.Value;
            _settings.ConfidenceThreshold = (int)_confidence.Value;
            _settings.ClassifyIntervalMs = (int)_interval.Value;
            _settings.SnapshotImageFormat = (SnapshotFormat)_snapshotFormat.SelectedIndex;
            _settings.RecordingFormat = (VideoFormat)_videoFormat.SelectedIndex;
            _settings.OutputDirectory = _outputDir.Text;

            DialogResult = DialogResult.OK;
            Close();
        }

        private static int Clamp(int value, int min, int max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }
    }
}
