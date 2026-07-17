// OnnxClassifier.cs - ONNX Runtime based image classifier for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.
//
// The Ultralytics YOLO26 detection model is exported end-to-end: it outputs a
// [1, N, 6] tensor whose rows are [x1, y1, x2, y2, confidence, classId] with
// non-maximum suppression already applied inside the model. Ultralytics stores
// the class names inside the ONNX model metadata (the "names" key), so no
// external label file is required.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ObjectDetector
{
    // A single classification result: a human readable label and a confidence
    // value between 0 and 1.
    public class Prediction
    {
        public string Label;
        public float Confidence;

        public Prediction(string label, float confidence)
        {
            Label = label;
            Confidence = confidence;
        }
    }

    // Wraps an ONNX Runtime inference session and performs preprocessing,
    // inference and post-processing for image classification.
    public class OnnxClassifier : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;
        private readonly int _inputWidth;
        private readonly int _inputHeight;
        private readonly int _detAttributes;
        private readonly Dictionary<int, string> _labels;

        public OnnxClassifier(string modelPath)
        {
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _session = new InferenceSession(modelPath, options);

            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            // Determine the model input resolution. Fixed dimensions are read
            // from the metadata; dynamic axes fall back to 640 (the export
            // resolution used by the accompanying Python script).
            int[] dims = _session.InputMetadata[_inputName].Dimensions;
            _inputHeight = (dims.Length >= 4 && dims[2] > 0) ? dims[2] : 640;
            _inputWidth = (dims.Length >= 4 && dims[3] > 0) ? dims[3] : 640;

            // The detection output has shape [1, N, attributes]; the last axis
            // holds the six values per detection (box, confidence, classId).
            int[] outDims = _session.OutputMetadata[_outputName].Dimensions;
            _detAttributes = (outDims.Length >= 1 && outDims[outDims.Length - 1] > 0)
                ? outDims[outDims.Length - 1]
                : 6;

            _labels = ReadLabels(_session);
        }

        // Runs the model on the given frame and returns the top predictions,
        // ordered from most to least confident.
        public List<Prediction> Classify(Bitmap frame, int topK)
        {
            DenseTensor<float> input = Preprocess(frame);

            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>();
            inputs.Add(NamedOnnxValue.CreateFromTensor(_inputName, input));

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs))
            {
                float[] data = results.First().AsTensor<float>().ToArray();

                // The end-to-end detection model outputs a flat [1, N, 6]
                // buffer. Each detection is a row of six floats:
                // [x1, y1, x2, y2, confidence, classId]. Column 5 is the class
                // index used to look up the object name; non-maximum suppression
                // is already applied inside the model.
                int cols = _detAttributes;
                List<Prediction> predictions = new List<Prediction>();
                if (cols >= 6 && data.Length >= cols)
                {
                    int rows = data.Length / cols;
                    for (int i = 0; i < rows; i++)
                    {
                        int offset = i * cols;
                        float confidence = data[offset + 4];
                        if (confidence <= 0f)
                        {
                            continue;
                        }

                        int classId = (int)data[offset + 5];
                        predictions.Add(new Prediction(GetLabel(classId), confidence));
                    }
                }

                return predictions
                    .OrderByDescending(p => p.Confidence)
                    .Take(topK)
                    .ToList();
            }
        }

        // Resizes the frame to the model input size and produces a normalised
        // CHW float tensor with pixel values scaled to the 0..1 range.
        private DenseTensor<float> Preprocess(Bitmap frame)
        {
            int planeSize = _inputWidth * _inputHeight;
            float[] buffer = new float[3 * planeSize];
            int rOffset = 0;
            int gOffset = planeSize;
            int bOffset = 2 * planeSize;

            using (Bitmap resized = new Bitmap(_inputWidth, _inputHeight, PixelFormat.Format24bppRgb))
            {
                using (Graphics g = Graphics.FromImage(resized))
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;
                    g.DrawImage(frame, 0, 0, _inputWidth, _inputHeight);
                }

                BitmapData data = resized.LockBits(
                    new Rectangle(0, 0, _inputWidth, _inputHeight),
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                try
                {
                    int stride = data.Stride;
                    unsafe
                    {
                        byte* scan0 = (byte*)data.Scan0.ToPointer();
                        int pixel = 0;
                        for (int y = 0; y < _inputHeight; y++)
                        {
                            byte* row = scan0 + (y * stride);
                            for (int x = 0; x < _inputWidth; x++)
                            {
                                // 24bpp bitmaps are stored as B, G, R. The tensor
                                // is laid out as three contiguous planes (CHW).
                                buffer[bOffset + pixel] = row[(x * 3) + 0] / 255f;
                                buffer[gOffset + pixel] = row[(x * 3) + 1] / 255f;
                                buffer[rOffset + pixel] = row[(x * 3) + 2] / 255f;
                                pixel++;
                            }
                        }
                    }
                }
                finally
                {
                    resized.UnlockBits(data);
                }
            }

            return new DenseTensor<float>(new Memory<float>(buffer), new int[] { 1, 3, _inputHeight, _inputWidth });
        }

        private string GetLabel(int index)
        {
            string label;
            if (_labels != null && _labels.TryGetValue(index, out label))
            {
                return label;
            }
            return "Class " + index.ToString(CultureInfo.InvariantCulture);
        }

        // Parses the Ultralytics "names" metadata entry, which looks like a
        // Python dictionary: {0: 'person', 1: 'bicycle', ...}.
        private static Dictionary<int, string> ReadLabels(InferenceSession session)
        {
            Dictionary<int, string> labels = new Dictionary<int, string>();
            try
            {
                ModelMetadata metadata = session.ModelMetadata;
                string names;
                if (metadata.CustomMetadataMap != null &&
                    metadata.CustomMetadataMap.TryGetValue("names", out names) &&
                    !string.IsNullOrEmpty(names))
                {
                    Regex regex = new Regex(@"(\d+)\s*:\s*'((?:[^'\\]|\\.)*)'");
                    foreach (Match m in regex.Matches(names))
                    {
                        int index;
                        if (int.TryParse(m.Groups[1].Value, out index))
                        {
                            string label = m.Groups[2].Value.Replace("\\'", "'").Replace("\\\\", "\\");
                            labels[index] = label;
                        }
                    }
                }
            }
            catch (Exception)
            {
                // Metadata is optional; fall back to generic class labels.
            }
            return labels;
        }

        public void Dispose()
        {
            if (_session != null)
            {
                _session.Dispose();
            }
        }
    }
}
