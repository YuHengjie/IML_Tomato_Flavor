# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'IML_flavor.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QPushButton,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(800, 720)
        self.label_title = QLabel(Form)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setGeometry(QRect(50, 10, 691, 61))
        font = QFont()
        font.setFamilies([u"Arial Black"])
        font.setPointSize(18)
        font.setBold(True)
        self.label_title.setFont(font)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_image_1 = QLabel(Form)
        self.label_image_1.setObjectName(u"label_image_1")
        self.label_image_1.setGeometry(QRect(169, 80, 610, 300))
        font1 = QFont()
        font1.setFamilies([u"Academy Engraved LET"])
        font1.setPointSize(18)
        self.label_image_1.setFont(font1)
        self.label_image_1.setAutoFillBackground(True)
        self.label_image_1.setAlignment(Qt.AlignCenter)
        self.layoutWidget = QWidget(Form)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(10, 80, 141, 601))
        self.verticalLayout_4 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_theme = QLabel(self.layoutWidget)
        self.label_theme.setObjectName(u"label_theme")

        self.verticalLayout_2.addWidget(self.label_theme)

        self.comboBox_theme = QComboBox(self.layoutWidget)
        self.comboBox_theme.setObjectName(u"comboBox_theme")

        self.verticalLayout_2.addWidget(self.comboBox_theme)


        self.verticalLayout_3.addLayout(self.verticalLayout_2)

        self.label_upload = QLabel(self.layoutWidget)
        self.label_upload.setObjectName(u"label_upload")

        self.verticalLayout_3.addWidget(self.label_upload)

        self.pushButton_upload = QPushButton(self.layoutWidget)
        self.pushButton_upload.setObjectName(u"pushButton_upload")

        self.verticalLayout_3.addWidget(self.pushButton_upload)


        self.verticalLayout_4.addLayout(self.verticalLayout_3)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_rating = QLabel(self.layoutWidget)
        self.label_rating.setObjectName(u"label_rating")

        self.verticalLayout.addWidget(self.label_rating)

        self.comboBox_rating = QComboBox(self.layoutWidget)
        self.comboBox_rating.setObjectName(u"comboBox_rating")

        self.verticalLayout.addWidget(self.comboBox_rating)

        self.label_scope = QLabel(self.layoutWidget)
        self.label_scope.setObjectName(u"label_scope")

        self.verticalLayout.addWidget(self.label_scope)

        self.comboBox_scope = QComboBox(self.layoutWidget)
        self.comboBox_scope.setObjectName(u"comboBox_scope")

        self.verticalLayout.addWidget(self.comboBox_scope)

        self.label_featureOrSample = QLabel(self.layoutWidget)
        self.label_featureOrSample.setObjectName(u"label_featureOrSample")

        self.verticalLayout.addWidget(self.label_featureOrSample)

        self.comboBox_featureOrSample = QComboBox(self.layoutWidget)
        self.comboBox_featureOrSample.setObjectName(u"comboBox_featureOrSample")

        self.verticalLayout.addWidget(self.comboBox_featureOrSample)


        self.verticalLayout_4.addLayout(self.verticalLayout)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.pushButton_run = QPushButton(self.layoutWidget)
        self.pushButton_run.setObjectName(u"pushButton_run")

        self.verticalLayout_5.addWidget(self.pushButton_run)

        self.label_output = QLabel(self.layoutWidget)
        self.label_output.setObjectName(u"label_output")

        self.verticalLayout_5.addWidget(self.label_output)

        self.textEdit_output = QTextEdit(self.layoutWidget)
        self.textEdit_output.setObjectName(u"textEdit_output")
        self.textEdit_output.setReadOnly(True)

        self.verticalLayout_5.addWidget(self.textEdit_output)


        self.verticalLayout_4.addLayout(self.verticalLayout_5)

        self.label_image_2 = QLabel(Form)
        self.label_image_2.setObjectName(u"label_image_2")
        self.label_image_2.setGeometry(QRect(169, 385, 610, 300))
        self.label_image_2.setFont(font1)
        self.label_image_2.setAutoFillBackground(True)
        self.label_image_2.setAlignment(Qt.AlignCenter)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label_title.setText(QCoreApplication.translate("Form", u"Interpretable machine learning-aided metabolomic selection \n"
"for enhanced tomato flavor", None))
        self.label_image_1.setText(QCoreApplication.translate("Form", u"Interpretation Plot: Zone One", None))
        self.label_theme.setText(QCoreApplication.translate("Form", u"Theme", None))
        self.label_upload.setText(QCoreApplication.translate("Form", u"Upload data", None))
        self.pushButton_upload.setText(QCoreApplication.translate("Form", u"Click", None))
        self.label_rating.setText(QCoreApplication.translate("Form", u"Consumer rating", None))
        self.label_scope.setText(QCoreApplication.translate("Form", u"Interpretation scope", None))
        self.label_featureOrSample.setText(QCoreApplication.translate("Form", u"Feature or sample", None))
        self.pushButton_run.setText(QCoreApplication.translate("Form", u"Run", None))
        self.label_output.setText(QCoreApplication.translate("Form", u"Output", None))
        self.label_image_2.setText(QCoreApplication.translate("Form", u"Interpretation Plot: Zone Two", None))
    # retranslateUi

