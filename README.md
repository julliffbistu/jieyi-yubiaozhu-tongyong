ģ���ϴ�ѹ������Ϊ�����֣�
1.modelMeta.json 
	˵���ϴ�ģ��Ԥ������������ݣ������������ѡ���
	json�е�keyֵ������ɾ��������������ݣ�����ʾΪ�ռ��ɡ�
	
	�����
		name:  ģ�����ƣ����磺�人��ӻ����񺽿ͻ�Ŀ��ʶ��
		weight��ģ��ѵ���Ľ��Ȩ�ز����ļ�,���磺airplane.th ��
	ѡ���
		dataset: ��¼��ģ��ѵ�������ݼ��������Ϣ��Ҫ�����������Ϣ
			name ����Ӧ����Ӣ�����ƣ����磺"background,airplane" 
			title��������������
			value������Ӧ��valueֵ
			color������Ӧ����ʾ��ɫ
			imageSize��ģ��ʹ�õ�ͼƬ��С
			bands��ģ�Ϳ����е�ͼƬ������
			dataType���������ͣ�������Byte,Uint16

		icon:��¼ͼƬ����,����ģ�����Ĳ�ͬ��ͼƬ���Ʋ�ͬ�����Ʋ��ɱ任
			����ָ
				image.png ԭʼͼƬ����
				label.png ��ǩͼƬ����
				pred.png  Ԥ��ͼƬ����
			Ŀ��ʶ��
				pred.png  Ԥ��ͼƬ����
			�仯��⣺
				A.png	  �仯ǰͼƬ����
				B.png	  �仯��ͼƬ����
				label.png ��ǩͼƬ����
				pred.png  Ԥ��ͼƬ����
		evaluate������ָ�꣬����json����ʽ������д
			accuracy: ׼ȷ�ȣ�Ҫ����дΪ �ٷֱ���ʽ������"90.3%"
			description��˵����Ӧָ����������
			best��˵����ģ�Ͷ�Ӧ������ָ����ֵ
			
			
	��ϸ��Ϣ�ɲ鿴modelMeta.json�ļ�
	�����д�淶���ϸ���json��ʽ
	
2.model�ļ���
	���ģ��Ԥ����뼰Ȩ���ļ�
	
	ע��1. Ȩ���ļ���������modelMeta.json����ȫһ��,�������"weight"�ļ����У�
		2. Ԥ��ģ��ͼƬ�����"icon"�ļ����У��������ͼƬչʾ����ɾ�����ļ��У�
		3. Ԥ���ļ�Predict.py��PredictModel.py���ļ����Ʋ��ɸ��ģ�
		4. Predict.py ģ������������ͼƬ������Ԥ�������ͼƬ�����沢�ɲ鿴��
			���������image_path(ͼƬ·��)��weight_path(Ȩ���ļ�·��)��gpu_num(ʹ��gpu����)
			���������ָ�仯��⣺ndarray����
				  Ŀ��ʶ��geojson ���� ���,����,��������
		5. PredictModel.py  ģ��Ԥ�⡣����ͼƬ·�����������Ԥ���������Ŀ��ʶ��Ԥ����geojson�ļ�
			���������image_path(ͼƬ·��)��weight_path(Ȩ���ļ�·��)��gpu_num(ʹ��gpu����)
			���������ָ�仯��⡢Ŀ��ʶ������ͼƬ
		6. Ԥ�����ṹ��ʵ����ʾ
		7. �ϴ�����ṹͼ������Ϊmodel.jpg, ������� /model�ļ�����,��������ṹͼƬ���������Ϊ��
		
3.ʹ��ģ����VAN/upernet_van_large_512x512_160k_ade20k.py
	basedir = os.path.abspath(os.path.dirname(__file__))
    weight_root = basedir + '/weight/'
    checkpoint_file = weight_root + 'latest.pth'
    config_file = r"configs/upernet/VAN/upernet_van_large_512x512_160k_ade20k.py"
    #checkpoint_file = r"tools/rsb128/latest.pth"