1. 将我们提供的models文件夹放在主目录下。训练好的模型是pth文件。
   ![image](https://github.com/user-attachments/assets/8d8213de-071c-440f-8388-7108bb383b81)

3. 效果测试
   首先，安装requirements.txt中所需要的库。然后，直接运行inference.py文件，可以测试效果。一开始运行时，会需要从远程的服务器下载一些基准模型，类似下图所示：
   <img width="868" alt="c9817442ce9d93ac8318bf1145ef55f" src="https://github.com/user-attachments/assets/be37730d-d01a-4b38-ae17-193c2f7217cd" />
   下载结束以后，会自动预测data/test_images.zip文件，并在同目录下生成结果。对于每一个zip文件，结果的结构如下图所示。其中，0、1、2、3分别表示点击率的4个分档。2和3表示高点击率分档。
   ![image](https://github.com/user-attachments/assets/656ab5bd-a977-4e7e-bd3d-b601ed30d487)
   同时还包含一个csv文件。test_images.zip文件的结果如下图所示。在使用前，先确认测试的结果与下图一致。
   ![image](https://github.com/user-attachments/assets/aa5f432e-4179-4387-83f0-ac1b1c836020)

4. 修改配置
   在运行新的测试文件前，需要先修改配置config/config_inference.yaml。修改zip_path，其表示zip文件的目录。如果是csv格式，需要同时指定csv文件，以及csv文件中的图片路径。
   ![image](https://github.com/user-attachments/assets/7239e652-7be1-4441-94c6-6829cae447d7)


   
