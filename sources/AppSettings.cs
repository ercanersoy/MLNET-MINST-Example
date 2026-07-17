// AppSettings.cs - Application settings model and persistence for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.

using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;

namespace ObjectDetector
{
    // Supported user interface languages.
    public enum AppLanguage
    {
        SystemDefault,
        Turkish,
        EnglishUK,
        EnglishUS
    }

    // Supported snapshot image formats.
    public enum SnapshotFormat
    {
        Bmp,
        Jpeg,
        Tiff,
        Png
    }

    // Supported video container formats. Only AVI is natively encoded without
    // an external library; the others are listed for completeness.
    public enum VideoFormat
    {
        Avi,
        Wmv,
        Mov,
        Mkv,
        Mpeg,
        Mpeg2,
        Mpeg4
    }

    // Holds every persisted user preference. Values are stored in a simple
    // "key=value" text file next to the executable.
    public class AppSettings
    {
        public AppLanguage Language;
        public int CameraIndex;
        public bool MirrorHorizontal;
        public int Brightness;              // -100 .. 100
        public int Contrast;                // -100 .. 100
        public int ConfidenceThreshold;     // 0 .. 100 (percent)
        public int ClassifyIntervalMs;      // milliseconds between classifications
        public SnapshotFormat SnapshotImageFormat;
        public VideoFormat RecordingFormat;
        public string OutputDirectory;

        public AppSettings()
        {
            Language = AppLanguage.SystemDefault;
            CameraIndex = 0;
            MirrorHorizontal = false;
            Brightness = 0;
            Contrast = 0;
            ConfidenceThreshold = 20;
            ClassifyIntervalMs = 500;
            SnapshotImageFormat = SnapshotFormat.Png;
            RecordingFormat = VideoFormat.Avi;
            OutputDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
        }

        // Returns the path of the settings file (next to the executable).
        public static string GetSettingsPath()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            return Path.Combine(baseDir, "settings.ini");
        }

        // Loads settings from disk, returning defaults if the file is missing
        // or unreadable.
        public static AppSettings Load()
        {
            AppSettings settings = new AppSettings();
            string path = GetSettingsPath();

            try
            {
                if (!File.Exists(path))
                {
                    return settings;
                }

                Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                string[] lines = File.ReadAllLines(path);
                foreach (string line in lines)
                {
                    string trimmed = line.Trim();
                    if (trimmed.Length == 0 || trimmed.StartsWith("#"))
                    {
                        continue;
                    }

                    int eq = trimmed.IndexOf('=');
                    if (eq <= 0)
                    {
                        continue;
                    }

                    string key = trimmed.Substring(0, eq).Trim();
                    string value = trimmed.Substring(eq + 1).Trim();
                    values[key] = value;
                }

                settings.Language = ParseEnum(values, "Language", AppLanguage.SystemDefault);
                settings.CameraIndex = ParseInt(values, "CameraIndex", 0);
                settings.MirrorHorizontal = ParseBool(values, "MirrorHorizontal", false);
                settings.Brightness = ParseInt(values, "Brightness", 0);
                settings.Contrast = ParseInt(values, "Contrast", 0);
                settings.ConfidenceThreshold = ParseInt(values, "ConfidenceThreshold", 20);
                settings.ClassifyIntervalMs = ParseInt(values, "ClassifyIntervalMs", 500);
                settings.SnapshotImageFormat = ParseEnum(values, "SnapshotImageFormat", SnapshotFormat.Png);
                settings.RecordingFormat = ParseEnum(values, "RecordingFormat", VideoFormat.Avi);

                string dir;
                if (values.TryGetValue("OutputDirectory", out dir) && dir.Length > 0)
                {
                    settings.OutputDirectory = dir;
                }
            }
            catch (Exception)
            {
                // Fall back to defaults on any read error.
                return new AppSettings();
            }

            return settings;
        }

        // Writes settings to disk. Errors are swallowed so that a read-only
        // location never crashes the application.
        public void Save()
        {
            try
            {
                List<string> lines = new List<string>();
                lines.Add("# Object Detector settings");
                lines.Add("Language=" + Language.ToString());
                lines.Add("CameraIndex=" + CameraIndex.ToString(CultureInfo.InvariantCulture));
                lines.Add("MirrorHorizontal=" + MirrorHorizontal.ToString());
                lines.Add("Brightness=" + Brightness.ToString(CultureInfo.InvariantCulture));
                lines.Add("Contrast=" + Contrast.ToString(CultureInfo.InvariantCulture));
                lines.Add("ConfidenceThreshold=" + ConfidenceThreshold.ToString(CultureInfo.InvariantCulture));
                lines.Add("ClassifyIntervalMs=" + ClassifyIntervalMs.ToString(CultureInfo.InvariantCulture));
                lines.Add("SnapshotImageFormat=" + SnapshotImageFormat.ToString());
                lines.Add("RecordingFormat=" + RecordingFormat.ToString());
                lines.Add("OutputDirectory=" + (OutputDirectory ?? string.Empty));

                File.WriteAllLines(GetSettingsPath(), lines.ToArray());
            }
            catch (Exception)
            {
                // Ignore write failures.
            }
        }

        // Returns a deep copy so dialogs can edit a temporary instance.
        public AppSettings Clone()
        {
            AppSettings copy = new AppSettings();
            copy.Language = Language;
            copy.CameraIndex = CameraIndex;
            copy.MirrorHorizontal = MirrorHorizontal;
            copy.Brightness = Brightness;
            copy.Contrast = Contrast;
            copy.ConfidenceThreshold = ConfidenceThreshold;
            copy.ClassifyIntervalMs = ClassifyIntervalMs;
            copy.SnapshotImageFormat = SnapshotImageFormat;
            copy.RecordingFormat = RecordingFormat;
            copy.OutputDirectory = OutputDirectory;
            return copy;
        }

        // Maps the snapshot format to a System.Drawing ImageFormat and file
        // extension.
        public static ImageFormat GetImageFormat(SnapshotFormat format, out string extension)
        {
            switch (format)
            {
                case SnapshotFormat.Bmp:
                    extension = ".bmp";
                    return ImageFormat.Bmp;
                case SnapshotFormat.Jpeg:
                    extension = ".jpg";
                    return ImageFormat.Jpeg;
                case SnapshotFormat.Tiff:
                    extension = ".tiff";
                    return ImageFormat.Tiff;
                default:
                    extension = ".png";
                    return ImageFormat.Png;
            }
        }

        public static string GetVideoExtension(VideoFormat format)
        {
            switch (format)
            {
                case VideoFormat.Avi: return ".avi";
                case VideoFormat.Wmv: return ".wmv";
                case VideoFormat.Mov: return ".mov";
                case VideoFormat.Mkv: return ".mkv";
                case VideoFormat.Mpeg: return ".mpeg";
                case VideoFormat.Mpeg2: return ".mpg";
                case VideoFormat.Mpeg4: return ".mp4";
                default: return ".avi";
            }
        }

        private static int ParseInt(Dictionary<string, string> values, string key, int fallback)
        {
            string raw;
            int result;
            if (values.TryGetValue(key, out raw) &&
                int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out result))
            {
                return result;
            }
            return fallback;
        }

        private static bool ParseBool(Dictionary<string, string> values, string key, bool fallback)
        {
            string raw;
            bool result;
            if (values.TryGetValue(key, out raw) && bool.TryParse(raw, out result))
            {
                return result;
            }
            return fallback;
        }

        private static T ParseEnum<T>(Dictionary<string, string> values, string key, T fallback)
        {
            string raw;
            if (values.TryGetValue(key, out raw))
            {
                try
                {
                    return (T)Enum.Parse(typeof(T), raw, true);
                }
                catch (Exception)
                {
                    return fallback;
                }
            }
            return fallback;
        }
    }
}
