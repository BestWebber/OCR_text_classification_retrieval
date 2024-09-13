# **OCR_text_classification_retrieval**
## **一、项目简介**
### **1. 项目背景**
这个项目服务于某企业的投资顾问部门，应用的场景是业务人员希望可以高效的利用本地大量的经济金融类书籍的正向信息为投资顾问场景提供真实可靠的理论依据和数据支撑。但是当前环境缺乏合适的pdf处理软件，且经济经融类书籍内容复杂，文本、图表、注释等经常混合分布，使用传统的pdf转化软件不能很好的处理。而且，即使得到处理后的文本如何高效的分类并添加到数据库中对于业务人员都存在较大难度。因此，该项目在云服务器上部署，搭建了云数据库、python虚拟环境，将代码部署好，业务人员直接在虚拟环境下调用py文件即可实现功能，非常轻便简洁，且易迁移。
### **2. 项目功能**
这个项目是部署在云服务器上的一个应用，可以同用户交互：上传需要处理的pdf文档、选择需要处理的页码范围；然后模型自动检测pdf模块并自动转化文本部分；接下来微调后的BERT模型自动完成文本分类并将正向文本添加在云服务器上搭建好的MySQL数据库中；最后用户可以通过该数据库实现检索。
### **3. 模块介绍**
由以下几个模块构成:
#### 1) data：主要存储数据
- material:存储上传的需要处理的文本
- input:打好标签的文本分类训练集数据
- output:中间表
#### 2）src：存储py文件
- OCR.py: 负责OCR模块，处理pdf文件，将结果存储在中间表中
- classify_model.py: fune_tune BERT模型完成文本情感分类任务
- model_prediction.py: 调用训练好的ckpt文件实现目标文本的预测并将正向文本添加至数据库
- query_retrival.py: 文本检索
#### 3）config：存储配置文件
- hfl--chinese-roberta-wwm-ext：huggingface下Roberta模型的预训练文件，作为无法从huggingface中直接下载的替代方案
- model_final.pth: detectron2模型路径文件，作为无法使用git clone detectron2的替代方案
- config.yaml: detectron2模型的配置文件
- saved_weights.ckpt: 保存classify_model.py的训练后的模型参数，用于预测时直接调用
#### 4）requirements.txt：存储需要下载的库及版本
#### 5）myenv：python虚拟环境
#### 6）detectron2: detectron2库
## **二、使用教程**
### **1. 登陆云服务器**
### **2. 使用`su -`并输入密码切换用户**
### **3. 进入目标文件夹**
`cd /marketing_story_generation`
### **4. 激活虚拟python系统**
`source myenv/bin/activate`
### **5. 从本地使用scp方法或者直接在云服务器可视化界面SFTP方式将需要处理的pdf文件上传到/material文件夹下**
`scp 'local_file_path' ubuntu@49.235.181.131:'remote_folder'`
其中`local_file_path`表示本地文件的路径，`remote_folder`表示将文件上传的目的路径。
例如，本地文件的存储路径为'/desktop/American_stock_investment.pdf', 目的地路径为'/marketing_story_generation/data/material/American_stock_investment.pdf', 则这行代码为：
`scp '/desktop/American_stock_investment.pdf'ubuntu@49.235.181.131:'/marketing_story_generation/data/material/American_stock_investment.pdf'`
> [!NOTE]
> 在上传时需要输入ubuntu系统的密码，且为了防止出现'No such file directory'的情况，建议将文件名修改为英文，并在全路径使用英文命名
### **6. 调用OCR.py完成pdf处理**
`cd /marketing_story_generation/src`
`python3 OCR.py`
#### 1）输入目的地路径选择需要处理的pdf文本
例如需要处理'/marketing_story_generation/data/material/American_stock_investment.pdf'，则在提示后输入该路径
#### 2）输入需要处理的范围
### **7. 如果需要重新训练文本情感分类模型**
`python3 classify_model.py`
> [!NOTE]
> 如果要执行该函数，建议需要考虑服务器的内存或者GPU的性能，另外如果没有GPU设备，训练速度会很慢
### **8. 对需要处理的文本分类并将积极文本上传至数据库**
`python3 model_prediction.py`
### **9. 从数据库中检索**
#### 1）运行函数
`python3 query_retrival.py`
#### 2）输入检索词
#### 3）返回检索结果

