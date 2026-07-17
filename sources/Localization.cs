// Localization.cs - User interface string tables for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;

namespace ObjectDetector
{
    // Provides translated user interface strings for the supported languages.
    // Strings are looked up by key; missing keys fall back to English (US).
    public static class Localization
    {
        private static AppLanguage _current = AppLanguage.SystemDefault;
        private static Dictionary<string, string> _active;

        private static readonly Dictionary<string, string> _turkish = BuildTurkish();
        private static readonly Dictionary<string, string> _englishUK = BuildEnglishUK();
        private static readonly Dictionary<string, string> _englishUS = BuildEnglishUS();

        static Localization()
        {
            SetLanguage(AppLanguage.SystemDefault);
        }

        public static AppLanguage Current
        {
            get { return _current; }
        }

        // Resolves "system default" to a concrete table based on the current
        // culture, then activates the matching string table.
        public static void SetLanguage(AppLanguage language)
        {
            _current = language;
            AppLanguage effective = language;

            if (effective == AppLanguage.SystemDefault)
            {
                effective = DetectSystemLanguage();
            }

            switch (effective)
            {
                case AppLanguage.Turkish:
                    _active = _turkish;
                    break;
                case AppLanguage.EnglishUK:
                    _active = _englishUK;
                    break;
                default:
                    _active = _englishUS;
                    break;
            }
        }

        // Returns the translated string for the given key.
        public static string Get(string key)
        {
            string value;
            if (_active != null && _active.TryGetValue(key, out value))
            {
                return value;
            }
            if (_englishUS.TryGetValue(key, out value))
            {
                return value;
            }
            return key;
        }

        // Formats a translated string that contains {0} style placeholders.
        public static string Format(string key, params object[] args)
        {
            return string.Format(CultureInfo.CurrentCulture, Get(key), args);
        }

        private static AppLanguage DetectSystemLanguage()
        {
            try
            {
                CultureInfo culture = Thread.CurrentThread.CurrentUICulture;
                string name = culture.Name;
                if (name.StartsWith("tr", StringComparison.OrdinalIgnoreCase))
                {
                    return AppLanguage.Turkish;
                }
                if (name.Equals("en-GB", StringComparison.OrdinalIgnoreCase))
                {
                    return AppLanguage.EnglishUK;
                }
            }
            catch (Exception)
            {
            }
            return AppLanguage.EnglishUS;
        }

        private static Dictionary<string, string> BuildTurkish()
        {
            Dictionary<string, string> d = new Dictionary<string, string>();
            d["App.Title"] = "Nesne Algılayıcı";
            d["Main.Detected"] = "Algılanan nesne:";
            d["Main.None"] = "(yok)";
            d["Main.History"] = "Nesne geçmişi";
            d["Main.StartCamera"] = "Kamerayı Başlat";
            d["Main.StopCamera"] = "Kamerayı Durdur";
            d["Main.Snapshot"] = "Anlık Görüntü";
            d["Main.StartRecord"] = "Kaydı Başlat";
            d["Main.StopRecord"] = "Kaydı Durdur";
            d["Main.Settings"] = "Ayarlar";
            d["Main.About"] = "Hakkında";
            d["Main.Recording"] = "● KAYDEDİLİYOR";
            d["Main.ClearHistory"] = "Geçmişi Temizle";
            d["Status.CameraStarted"] = "Kamera başlatıldı.";
            d["Status.CameraStopped"] = "Kamera durduruldu.";
            d["Status.SnapshotSaved"] = "Anlık görüntü kaydedildi: {0}";
            d["Status.RecordingStarted"] = "Video kaydı başladı: {0}";
            d["Status.RecordingStopped"] = "Video kaydı durduruldu: {0}";
            d["Status.ModelLoaded"] = "Model yüklendi: {0}";
            d["Confidence"] = "%{0} güven";
            d["Error.Title"] = "Hata";
            d["Error.ModelNotFound"] = "Model dosyası bulunamadı:\n{0}\n\nÖnce build.bat ile modeli üretin.";
            d["Error.ModelLoad"] = "Model yüklenemedi:\n{0}";
            d["Error.CameraStart"] = "Kamera başlatılamadı. Bir video yakalama aygıtı bağlı olduğundan emin olun.";
            d["Error.NoFrame"] = "Kaydedilecek görüntü yok. Önce kamerayı başlatın.";
            d["Error.SnapshotSave"] = "Anlık görüntü kaydedilemedi:\n{0}";
            d["Error.RecordStart"] = "Video kaydı başlatılamadı:\n{0}";
            d["Error.VideoFormatUnsupported"] = "Seçilen video biçimi ({0}) harici kütüphane olmadan desteklenmiyor. Yalnızca AVI biçimi native olarak kaydedilir. Lütfen ayarlardan AVI seçin.";
            d["Settings.Title"] = "Ayarlar";
            d["Settings.Language"] = "Dil";
            d["Settings.LanguageGroup"] = "Dil Ayarları";
            d["Settings.ImageGroup"] = "Görüntü Ayarları";
            d["Settings.OutputGroup"] = "Çıktı Ayarları";
            d["Settings.DetectionGroup"] = "Algılama Ayarları";
            d["Settings.Camera"] = "Kamera aygıtı (dizin)";
            d["Settings.Mirror"] = "Görüntüyü yatay aynala";
            d["Settings.Brightness"] = "Parlaklık";
            d["Settings.Contrast"] = "Karşıtlık (kontrast)";
            d["Settings.Confidence"] = "Güven eşiği (%)";
            d["Settings.Interval"] = "Algılama aralığı (ms)";
            d["Settings.SnapshotFormat"] = "Anlık görüntü biçimi";
            d["Settings.VideoFormat"] = "Video biçimi";
            d["Settings.OutputDir"] = "Çıktı klasörü";
            d["Settings.Browse"] = "Gözat...";
            d["Settings.Ok"] = "Tamam";
            d["Settings.Cancel"] = "İptal";
            d["Lang.System"] = "Sistem varsayılanı";
            d["Lang.Turkish"] = "Türkçe";
            d["Lang.EnglishUK"] = "İngilizce (Birleşik Krallık)";
            d["Lang.EnglishUS"] = "İngilizce (Amerika Birleşik Devletleri)";
            d["About.Title"] = "Hakkında";
            d["About.Description"] = "Ultralytics YOLO26 (ONNX) modeli ile nesne algılama uygulaması.";
            d["About.Version"] = "Sürüm 1.0";
            d["About.Copyright"] = "Telif Hakkı (c) 2026 Ercan Ersoy";
            d["About.License"] = "MIT Lisansı ile lisanslanmıştır.";
            d["About.Ok"] = "Tamam";
            return d;
        }

        private static Dictionary<string, string> BuildEnglishUS()
        {
            Dictionary<string, string> d = new Dictionary<string, string>();
            d["App.Title"] = "Object Detector";
            d["Main.Detected"] = "Detected object:";
            d["Main.None"] = "(none)";
            d["Main.History"] = "Object history";
            d["Main.StartCamera"] = "Start Camera";
            d["Main.StopCamera"] = "Stop Camera";
            d["Main.Snapshot"] = "Snapshot";
            d["Main.StartRecord"] = "Start Recording";
            d["Main.StopRecord"] = "Stop Recording";
            d["Main.Settings"] = "Settings";
            d["Main.About"] = "About";
            d["Main.Recording"] = "● RECORDING";
            d["Main.ClearHistory"] = "Clear History";
            d["Status.CameraStarted"] = "Camera started.";
            d["Status.CameraStopped"] = "Camera stopped.";
            d["Status.SnapshotSaved"] = "Snapshot saved: {0}";
            d["Status.RecordingStarted"] = "Recording started: {0}";
            d["Status.RecordingStopped"] = "Recording stopped: {0}";
            d["Status.ModelLoaded"] = "Model loaded: {0}";
            d["Confidence"] = "{0}% confidence";
            d["Error.Title"] = "Error";
            d["Error.ModelNotFound"] = "Model file not found:\n{0}\n\nBuild the model first with build.bat.";
            d["Error.ModelLoad"] = "Failed to load the model:\n{0}";
            d["Error.CameraStart"] = "Failed to start the camera. Make sure a video capture device is connected.";
            d["Error.NoFrame"] = "There is no frame to save. Start the camera first.";
            d["Error.SnapshotSave"] = "Failed to save the snapshot:\n{0}";
            d["Error.RecordStart"] = "Failed to start recording:\n{0}";
            d["Error.VideoFormatUnsupported"] = "The selected video format ({0}) is not supported without an external library. Only AVI is encoded natively. Please choose AVI in the settings.";
            d["Settings.Title"] = "Settings";
            d["Settings.Language"] = "Language";
            d["Settings.LanguageGroup"] = "Language Settings";
            d["Settings.ImageGroup"] = "Image Settings";
            d["Settings.OutputGroup"] = "Output Settings";
            d["Settings.DetectionGroup"] = "Detection Settings";
            d["Settings.Camera"] = "Camera device (index)";
            d["Settings.Mirror"] = "Mirror image horizontally";
            d["Settings.Brightness"] = "Brightness";
            d["Settings.Contrast"] = "Contrast";
            d["Settings.Confidence"] = "Confidence threshold (%)";
            d["Settings.Interval"] = "Detection interval (ms)";
            d["Settings.SnapshotFormat"] = "Snapshot format";
            d["Settings.VideoFormat"] = "Video format";
            d["Settings.OutputDir"] = "Output folder";
            d["Settings.Browse"] = "Browse...";
            d["Settings.Ok"] = "OK";
            d["Settings.Cancel"] = "Cancel";
            d["Lang.System"] = "System default";
            d["Lang.Turkish"] = "Turkish";
            d["Lang.EnglishUK"] = "English (United Kingdom)";
            d["Lang.EnglishUS"] = "English (United States)";
            d["About.Title"] = "About";
            d["About.Description"] = "An object detection application using the Ultralytics YOLO26 (ONNX) model.";
            d["About.Version"] = "Version 1.0";
            d["About.Copyright"] = "Copyright (c) 2026 Ercan Ersoy";
            d["About.License"] = "Licensed under the MIT License.";
            d["About.Ok"] = "OK";
            return d;
        }

        // British English differs only in spelling from US English; start from
        // the US table and override the relevant strings.
        private static Dictionary<string, string> BuildEnglishUK()
        {
            Dictionary<string, string> d = new Dictionary<string, string>(BuildEnglishUS());
            d["App.Title"] = "Object Detector";
            d["Settings.OutputGroup"] = "Output Settings";
            d["About.License"] = "Licensed under the MIT Licence.";
            d["Lang.EnglishUK"] = "English (United Kingdom)";
            return d;
        }
    }
}
