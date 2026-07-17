// AboutForm.cs - "About" dialog for Object Detector
//
// Copyright (c) 2026 Ercan Ersoy.
// This file is licensed under the MIT License.
// Written by Ercan Ersoy helped by Claude Opus 4.8.

using System;
using System.Drawing;
using System.Windows.Forms;

namespace ObjectDetector
{
    public class AboutForm : Form
    {
        public AboutForm()
        {
            Text = Localization.Get("About.Title");
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition = FormStartPosition.CenterParent;
            MaximizeBox = false;
            MinimizeBox = false;
            ClientSize = new Size(420, 220);

            Label title = new Label();
            title.Text = Localization.Get("App.Title");
            title.Font = new Font(Font.FontFamily, 14f, FontStyle.Bold);
            title.AutoSize = true;
            title.Location = new Point(20, 20);

            Label version = new Label();
            version.Text = Localization.Get("About.Version");
            version.AutoSize = true;
            version.Location = new Point(20, 56);

            Label description = new Label();
            description.Text = Localization.Get("About.Description");
            description.AutoSize = false;
            description.Location = new Point(20, 84);
            description.Size = new Size(380, 40);

            Label copyright = new Label();
            copyright.Text = Localization.Get("About.Copyright");
            copyright.AutoSize = true;
            copyright.Location = new Point(20, 132);

            Label license = new Label();
            license.Text = Localization.Get("About.License");
            license.AutoSize = true;
            license.Location = new Point(20, 156);

            Button okButton = new Button();
            okButton.Text = Localization.Get("About.Ok");
            okButton.DialogResult = DialogResult.OK;
            okButton.Size = new Size(90, 28);
            okButton.Location = new Point(310, 180);

            Controls.Add(title);
            Controls.Add(version);
            Controls.Add(description);
            Controls.Add(copyright);
            Controls.Add(license);
            Controls.Add(okButton);

            AcceptButton = okButton;
        }
    }
}
