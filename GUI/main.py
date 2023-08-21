from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
from Ui_IML_flavor import Ui_Form
from PySide6 import QtGui,QtCore

import qtmodern.styles
import qtmodern.windows

from matplotlib.backends.backend_qtagg import FigureCanvas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

import joblib
import shap
import lime
import lime.lime_tabular

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

class MyWidget(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()  
        self.setupUi(self) 
        self.setWindowTitle("IML for tomato flavor")
        
        self.comboBox_theme.addItems(['Light', 'Dark']) 
        self.comboBox_theme.setCurrentText("Light")  
        self.selected_theme = 'Light'  
        self.comboBox_theme.currentTextChanged.connect(self.change_theme)  

        self.dataset = pd.DataFrame()
        self.pushButton_upload.clicked.connect(self.upload_xlsx)  

        self.comboBox_rating.addItems(['Overall liking', 'Sweetness', 'Sourness', 'Umami', 'Flavor intensity'])  # Add "Light" option
        self.comboBox_rating.setCurrentText('Overall liking') 
        self.selected_rating = 'Overall liking'  
        self.comboBox_rating.currentTextChanged.connect(self.set_selected_rating)  

        self.comboBox_scope.addItems(['Global', 'Local']) 
        self.comboBox_scope.setCurrentText('Global')  
        self.selected_scope = 'Global'  
        self.comboBox_scope.currentTextChanged.connect(self.set_selected_scope)  

        self.feature_selected = pd.read_excel("./Final_selected_features.xlsx",index_col = 0,)
        self.selected_feature_sample = ''
        self.update_feature_sample_items() 
        self.comboBox_featureOrSample.currentTextChanged.connect(self.set_selected_feature_sample)

        self.pushButton_run.clicked.connect(self.show_plot)

        self.label_image_1.setStyleSheet("background-color: #DCDCDC;")
        self.label_image_2.setStyleSheet("background-color: #DCDCDC;")
        self.textEdit_output.setStyleSheet("background-color: #F5F5F5;")

        self.lable_image_status = 0

    def upload_xlsx(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open XLSX File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_path:
            self.dataset = pd.read_excel(file_path,index_col=0)
        if len(self.dataset) != 0:
            self.textEdit_output.append("Data uploaded.")
        if self.selected_scope == "Local":
            self.update_feature_sample_items()

    def change_theme(self, style_name):
        if style_name == "Light":
            qtmodern.styles.light(QApplication.instance())
            self.label_image_1.setStyleSheet("background-color: #DCDCDC;")
            self.label_image_2.setStyleSheet("background-color: #DCDCDC;")
            self.textEdit_output.setStyleSheet("background-color: #F5F5F5;")
        elif style_name == "Dark":
            qtmodern.styles.dark(QApplication.instance())
            self.label_image_1.setStyleSheet("background-color: #696969;")
            self.label_image_2.setStyleSheet("background-color: #696969;")
            self.textEdit_output.setStyleSheet("background-color: #323232;")
        self.selected_theme = style_name
        self.textEdit_output.append(f"Theme selected: {self.selected_theme}.")
        if self.lable_image_status == 1:
            self.textEdit_output.append("Click run to update plot theme.")

    def set_selected_rating(self, rating):
        self.selected_rating = rating
        self.textEdit_output.append(f"Consumer rating selected: {self.selected_rating}.")
        self.update_feature_sample_items()

    def set_selected_scope(self, scope):
        self.selected_scope = scope
        self.textEdit_output.append(f"Interpretation scope selected: {self.selected_scope}.")
        self.update_feature_sample_items()

    def set_selected_feature_sample(self, feature_sample):
        self.selected_feature_sample = feature_sample
        if self.selected_scope == "Local" and len(self.dataset) != 0:
            self.textEdit_output.append(f"Sample selected: {self.selected_feature_sample}.")
            self.textEdit_output.append("Click run to get plots !")
        elif self.selected_scope == "Global":
            self.textEdit_output.append(f"Feature selected: {self.selected_feature_sample}.")
            self.textEdit_output.append("Click run to get plots !")

    def update_feature_sample_items(self):
        if self.selected_scope == 'Global':
            feature_sample_items = self.feature_selected.loc[self.selected_rating, 'Selected features']
            feature_sample_list = ast.literal_eval(feature_sample_items)
        elif self.selected_scope == 'Local':
            if len(self.dataset) == 0:
                self.textEdit_output.append("Please upload data file first !")
            feature_sample_list = self.dataset.index.tolist()
        if self.selected_feature_sample == '': 
            self.selected_feature_sample = feature_sample_list[0]
        self.comboBox_featureOrSample.clear()
        self.comboBox_featureOrSample.addItems(feature_sample_list)

    def canvas_to_qimage(self,canvas):
        canvas.draw()
        width, height = canvas.get_width_height()
        image = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)
        return image.copy()
    
    def obtain_shap_values(self):
        y_name = self.selected_rating
        shap_values_all = np.load("./SHAP/shap_values_{}.npy".format(y_name))
        X_test_all = pd.read_excel("./SHAP/X_test_{}.xlsx".format(y_name),index_col = 0,)
        return shap_values_all,X_test_all
    
    def plot_shap_global(self):
        feature_name_list = self.feature_selected.loc[self.selected_rating, 'Selected features']
        feature_name_list = ast.literal_eval(feature_name_list)
        index = feature_name_list.index(self.selected_feature_sample)
        shap_values_all,X_test_all = self.obtain_shap_values()
        global_main_effects = shap_values_all[:,index]

        fig, ax = plt.subplots(figsize=(6.1,3))
        ax.scatter(X_test_all[self.selected_feature_sample], global_main_effects, marker='o',s=45,c='#1E90FF',linewidth=0.2,edgecolors='#FFFFFF',alpha=0.7)
        ax.set_xlabel(self.selected_feature_sample) 
        ax.set_ylabel('SHAP value')  

        if self.selected_theme == 'Light':
            fig.set_facecolor('#F5F5F5')
            ax.set_facecolor('#F5F5F5')
        elif self.selected_theme == 'Dark':
            fig.set_facecolor('#C0C0C0')
            ax.set_facecolor('#C0C0C0')

        plt.tight_layout()
        return fig
    
    def plot_pdp_ice(self):
        ICE_lines_show = pd.read_csv("./PDP_ICE/ICE_{}_{}.zip".format(self.selected_rating,self.selected_feature_sample),index_col=0)
        PDP_lines_df = pd.read_csv("./PDP_ICE/PDP_{}_{}.zip".format(self.selected_rating,self.selected_feature_sample),index_col=0)
        ICE_lines_show.columns = [float(col) for col in ICE_lines_show.columns]

        fig, ax = plt.subplots(figsize=(6.1, 3))
        for i in range(len(ICE_lines_show)):
            pd_temp = ICE_lines_show.iloc[i,:].dropna()
            ax.plot(pd_temp.index,pd_temp.values,linewidth=0.2,color='#696969',zorder=-1)

        ax.plot(PDP_lines_df['x_mean'], PDP_lines_df['y_mean'], marker='o',markersize=5,label='Average PDP',zorder=2,linewidth=3,color='#1E90FF')
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel(self.selected_feature_sample) 
        ax.set_ylabel('Predicted probability')
        
        if self.selected_theme == 'Light':
            fig.set_facecolor('#F5F5F5')
            ax.set_facecolor('#F5F5F5')
        elif self.selected_theme == 'Dark':
            fig.set_facecolor('#C0C0C0')
            ax.set_facecolor('#C0C0C0')
            
        plt.tight_layout()
        return fig

    def series_to_string_with_equals(self,series):
        series_dict = series.to_dict()
        result_string = [f"{str(index)} = {str(value)}" for index, value in series_dict.items()]
        return result_string

    def plot_shap_local(self):
        y_name = self.selected_rating
        index = self.selected_feature_sample
        feature_sample_items = self.feature_selected.loc[y_name, 'Selected features']
        feature_sample_list = ast.literal_eval(feature_sample_items)
        data_df = self.dataset.loc[index,feature_sample_list]
        shap_df = pd.DataFrame(index=range(1,11,1),columns=feature_sample_list)

        for i in range(1,11,1):
            model = joblib.load("./Model/model_{}_{}.pkl".format(y_name, i))
            explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(data_df)[1]
            shap_df.loc[i,:] = shap_values

        mean_row = pd.DataFrame(shap_df.mean(axis=0).values.reshape(1, -1), columns=shap_df.columns, index=['Mean'])
        shap_df = pd.concat([shap_df,mean_row])

        plot_df = pd.DataFrame(index=range(1,len(feature_sample_list)+1,1),columns=['Ticks','SHAP value'])
        plot_df['Ticks'] = self.series_to_string_with_equals(data_df)
        plot_df['SHAP value'] = shap_df.loc['Mean',:].values
        plot_df['abs SHAP value'] = abs(plot_df['SHAP value'])
        plot_df = plot_df.sort_values(by='abs SHAP value')
        plot_df = plot_df.iloc[-8:,:]
        plot_df = plot_df.sort_values(by='SHAP value')

        bars = plot_df['Ticks'].values
        height = plot_df['SHAP value'].values

        colors = []
        for i in range(0,len(bars)):
            if height[i] > 0:
                colors.append('#008BFA')
            else:
                colors.append('#FF0050')

        fig, ax = plt.subplots(figsize=(6.1, 3))

        sns.set_style("whitegrid")
        ax.margins(0.05)
        ax.grid(linestyle=(0, (1, 6.5)), color='#B0B0B0', zorder=0)
        ax.barh(range(0, len(bars)), height, color=colors, edgecolor="none", zorder=3)
        ax.set_yticks(range(0, len(bars)))
        ax.set_yticklabels(bars)
        ax.set_xlabel('SHAP value')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)

        ax.tick_params(top=False,
                    bottom=True,
                    left=False,
                    right=False)
        
        if self.selected_theme == 'Light':
            fig.set_facecolor('#F5F5F5')
            ax.set_facecolor('#F5F5F5')
        elif self.selected_theme == 'Dark':
            fig.set_facecolor('#C0C0C0')
            ax.set_facecolor('#C0C0C0')

        plt.tight_layout()
        return fig
    
    def plot_lime(self):
        y_name = self.selected_rating
        index = self.selected_feature_sample
        feature_sample_items = self.feature_selected.loc[y_name, 'Selected features']
        feature_sample_list = ast.literal_eval(feature_sample_items)
        data_df = self.dataset.loc[index,feature_sample_list]
        dataset_explain = self.dataset.loc[:,feature_sample_list]

        model_best_dict = {'Overall liking':1,
                        'Sweetness':4,
                        'Sourness':2,
                        'Umami':7,
                        'Flavor intensity':6,}

        model = joblib.load("./Model/model_{}_{}.pkl".format(y_name, model_best_dict[y_name]))
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(dataset_explain.values), 
                                                            feature_names=dataset_explain.columns,
                                                            discretize_continuous=True,random_state=42)
        exp = explainer.explain_instance(data_df, model.predict_proba,
                                        num_features=8)
        lime_list = exp.as_list()

        ticks_column = [item[0] for item in lime_list]
        value_column = [item[1] for item in lime_list]
        lime_df = pd.DataFrame(index=range(0,len(lime_list)),columns=['Ticks','LIME value'])
        lime_df['Ticks'] = ticks_column
        lime_df['LIME value'] = value_column

        plot_df = lime_df.sort_values(by='LIME value')

        bars = plot_df['Ticks'].values
        height = plot_df['LIME value'].values

        colors = []
        for i in range(0,len(bars)):
            if height[i] > 0:
                colors.append('#008BFA')
            else:
                colors.append('#FF0050')

        fig, ax = plt.subplots(figsize=(6.1, 3))

        plt.style.use('seaborn-ticks')
        ax.margins(0.05)
        ax.grid(linestyle=(0, (1, 6.5)), color='#B0B0B0', zorder=0)
        ax.barh(range(0, len(bars)), height, color=colors, edgecolor="none", zorder=3)
        ax.set_yticks(range(0, len(bars)))
        ax.set_yticklabels(bars)
        ax.set_xlabel('LIME value')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)

        ax.tick_params(top=False,
                    bottom=True,
                    left=False,
                    right=False)
        
        if self.selected_theme == 'Light':
            fig.set_facecolor('#F5F5F5')
            ax.set_facecolor('#F5F5F5')
        elif self.selected_theme == 'Dark':
            fig.set_facecolor('#C0C0C0')
            ax.set_facecolor('#C0C0C0')

        plt.tight_layout()
        return fig
    
    
    def show_plot(self):
        self.lable_image_status = 1
        if self.selected_feature_sample =='':
            self.update_feature_sample_items()
        if self.selected_scope == 'Global':
            plot = self.plot_shap_global()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_1.setPixmap(QtGui.QPixmap.fromImage(image))
            self.label_image_1.setAlignment(QtCore.Qt.AlignCenter)

            plot = self.plot_pdp_ice()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_2.setPixmap(QtGui.QPixmap.fromImage(image))

            self.textEdit_output.append("Global interpretation successful !")

        elif self.selected_scope == 'Local':
            plot = self.plot_shap_local()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_1.setPixmap(QtGui.QPixmap.fromImage(image))
            self.label_image_1.setAlignment(QtCore.Qt.AlignCenter)

            plot = self.plot_lime()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_2.setPixmap(QtGui.QPixmap.fromImage(image))
            self.label_image_2.setAlignment(QtCore.Qt.AlignCenter)
            self.textEdit_output.append("Local interpretation successful !")

def show_plot_in_theme_change(self):
        self.lable_image_status = 1
        if self.selected_feature_sample =='':
            self.update_feature_sample_items()
        if self.selected_scope == 'Global':
            plot = self.plot_shap_global()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_1.setPixmap(QtGui.QPixmap.fromImage(image))
            self.label_image_1.setAlignment(QtCore.Qt.AlignCenter)

            plot = self.plot_pdp_ice()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_2.setPixmap(QtGui.QPixmap.fromImage(image))

        elif self.selected_scope == 'Local':
            plot = self.plot_shap_local()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_1.setPixmap(QtGui.QPixmap.fromImage(image))
            self.label_image_1.setAlignment(QtCore.Qt.AlignCenter)

            plot = self.plot_lime()
            canvas = FigureCanvas(plot)
            image = self.canvas_to_qimage(canvas)  
            self.label_image_2.setPixmap(QtGui.QPixmap.fromImage(image))
            self.label_image_2.setAlignment(QtCore.Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication()
    widget = MyWidget()
    qtmodern.styles.light(app) # dark
    mw = qtmodern.windows.ModernWindow(widget)
    mw.show()
    widget.show()
    app.exec()
