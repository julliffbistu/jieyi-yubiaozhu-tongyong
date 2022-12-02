模型上传压缩包分为两部分：
1.modelMeta.json 
	说明上传模型预测代码的相关内容，包括必填项和选填项。
	json中的key值，不可删除，若无相关数据，则显示为空即可。
	
	必填项：
		name:  模型名称，例如：武汉天河机场民航客机目标识别
		weight：模型训练的结果权重参数文件,例如：airplane.th 。
	选填项：
		dataset: 记录改模型训练的数据集的相关信息，要求包括背景信息
			name ：对应类别的英文名称，例如："background,airplane" 
			title：类别的中文名称
			value：类别对应的value值
			color：类别对应的显示颜色
			imageSize：模型使用的图片大小
			bands：模型可运行的图片波段数
			dataType：数据类型，包括：Byte,Uint16

		icon:记录图片名称,根据模型类别的不同，图片名称不同，名称不可变换
			语义分割：
				image.png 原始图片名称
				label.png 标签图片名称
				pred.png  预测图片名称
			目标识别：
				pred.png  预测图片名称
			变化检测：
				A.png	  变化前图片名称
				B.png	  变化后图片名称
				label.png 标签图片名称
				pred.png  预测图片名称
		evaluate：评价指标，按照json中样式进行填写
			accuracy: 准确度，要求填写为 百分比样式，例如"90.3%"
			description：说明对应指标中文名称
			best：说明改模型对应的评价指标数值
			
			
	详细信息可查看modelMeta.json文件
	相关书写规范需严格按照json格式
	
2.model文件夹
	存放模型预测代码及权重文件
	
	注：1. 权重文件名称需与modelMeta.json中完全一致,并存放在"weight"文件夹中；
		2. 预测模型图片存放在"icon"文件夹中，若无相关图片展示，则删除改文件夹；
		3. 预测文件Predict.py、PredictModel.py，文件名称不可更改；
		4. Predict.py 模型评估。输入图片，进行预测后生成图片，保存并可查看；
			输入参数：image_path(图片路径)、weight_path(权重文件路径)、gpu_num(使用gpu个数)
			输出：语义分割、变化检测：ndarray数组
				  目标识别：geojson 包括 类别,概率,矩形坐标
		5. PredictModel.py  模型预测。输入图片路径，输出的是预测后的数组或目标识别预测后的geojson文件
			输入参数：image_path(图片路径)、weight_path(权重文件路径)、gpu_num(使用gpu个数)
			输出：语义分割、变化检测、目标识别：生成图片
		6. 预测代码结构如实例所示
		7. 上传网络结构图，名称为model.jpg, 并存放在 /model文件夹下,若无网络结构图片，则可设置为空
		
3.使用模型是VAN/upernet_van_large_512x512_160k_ade20k.py
	basedir = os.path.abspath(os.path.dirname(__file__))
    weight_root = basedir + '/weight/'
    checkpoint_file = weight_root + 'latest.pth'
    config_file = r"configs/upernet/VAN/upernet_van_large_512x512_160k_ade20k.py"
    #checkpoint_file = r"tools/rsb128/latest.pth"