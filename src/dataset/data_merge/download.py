import os
import shutil
import hashlib
import uuid
import subprocess
import zipfile
import json

import tensorflow as tf

from collections import OrderedDict

def extract_compression_file(fileName, target_folder):
    '''
    Extract zip file and remove system files
    '''

    if not os.path.isdir('temp'):
        os.mkdir('temp')
    if '.zip' == os.path.splitext(fileName)[-1].lower():
        zip_ref = zipfile.ZipFile(fileName, 'r')
        zip_ref.extractall('temp')
        zip_ref.close()
    else:
        raise ValueError('Date compression File should be "Zip" file')

    folders = os.listdir('temp')
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    for fold in folders:
        if fold == '__MACOSX' or fold.startswith('.'):
            shutil.rmtree(os.path.join('temp', fold))
        else:
            dirPath = os.path.join(target_folder, fold)
            shutil.move(os.path.join('temp', fold), dirPath)
    shutil.rmtree('temp')


# def is_same_image(image_path1, image_path2):
#     hash_md5 = hashlib.md5()
#     with open(image_path1, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#     md5_1 = hash_md5.hexdigest()

#     hash_md5_2 = hashlib.md5()
#     with open(image_path2, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5_2.update(chunk)
#     md5_2 = hash_md5_2.hexdigest()

#     if md5_1 == md5_2:
#         return True
#     else:
#         return False

def merge_metadata(src_metadata_list,dst_metadata_list):
    '''
    src_metadata_list: Source metadata, need to merge
    dst_metadata_list: Destination metadata

    Both src_metadata_list and dst_metadata_list should follow the format such as below:

    [{'ID':1,'Name': [The Class Name]},{'ID':2,'Name': [The Class Name]},{'ID':3,'Name': [The Class Name]}...]

    or

    [{'ID':1,'Name': [The Class Name]},{'ID':3,'Name': [The Class Name]}...]

    '''
    src_dict = {}
    dst_dict = {}

    if src_metadata_list is None:
        # If src dosen't have metadata.json file
        raise ValueError('Please give this dataset "metadata.json" file')
    else:
        for src_metadata in src_metadata_list:
            if src_metadata['ID'] not in src_dict.keys():
                src_dict[src_metadata['ID']]=src_metadata['Name']
    
    if dst_metadata_list is None: 
        tf.logging.info('It is the first time to merge data')
    else:
        for dst_metadata in dst_metadata_list:
            if dst_metadata['ID'] not in dst_dict.keys():
                dst_dict[dst_metadata['ID']]=dst_metadata['Name']


    for src_key,src_value in src_dict.items():
        if src_key in dst_dict.keys():
            if src_value == dst_dict[src_key]:
                # If src key and value are in dst, just merge it directly. E.g. src = { "1" : A, "2":B}, dst = {"1":A, "2":B,...}
                tf.logging.info('Merge Class: ID = [{}] , Name = [{}]'.format(src_key,src_value))
            else:
                # If src key in dst, but src value mismatch dst value which has that key, will raise Error. E.g. src = { "1" : A, "2":C}, dst = {"1":A, "2":B,...}
                raise ValueError('Src Class ID [{}] and Name [{}] is not match with dst Class ID [{}] and Name [{}]'.format(src_key,src_value,src_key,dst_dict[src_key]))
        elif src_value in dst_dict.values():
            # If src key not in dst, but src value which belongs to the key is also in dst, which means class name is the same, but class number is not
            # E.g. src = { "1" : A, "3":B}, dst = {"1":A, "2":B,...}
            raise ValueError('Src Class Name [{}] is in dst Class which has ID [{}], but not match with dst Class ID'.format(src_value,src_key))
        else:
            # New src key and value
            dst_dict[src_key]=src_value
            tf.logging.info('Update Class: ID = [{}] , Name = [{}]'.format(src_key,src_value))
    
    dst_dict = OrderedDict(sorted(dst_dict.items(), key=lambda t: t[0],reverse=False))

    merged_dst_list = []

    class_num = len(dst_dict.keys())
    # Create the merged meta_data_list
    for k,v in dst_dict.items():
        merged_dst_list.append({'ID':k,'Name':v})

    return merged_dst_list,class_num


def get_metadata_info(path):
    with open(path,"r") as data_info:
            metadata_list = json.load(data_info)['items']
    return metadata_list

def write_metadata_info(path,metadata_list):
    update_metadata = {'items':metadata_list}
    with open(path,"w") as data_info:
        json.dump(update_metadata,data_info)



def merger_folder(merge_zip_list, target_folder_name):
    # Segmentation specified
    if not os.path.isdir(os.path.join(target_folder_name,'Image')): 
        os.makedirs(os.path.join(target_folder_name,'Image'))
        target_image_folder = os.path.join(target_folder_name,'Image')
    if not os.path.isdir(os.path.join(target_folder_name,'Mask')): 
        os.makedirs(os.path.join(target_folder_name,'Mask'))
        target_mask_folder = os.path.join(target_folder_name,'Mask')

    dst_metadata_list = []
    if 'metadata.json' in os.listdir(target_folder_name):
        dst_metadata_list = get_metadata_info(os.path.join(target_folder_name,'metadata.json'))
    
    image_cnt = 0
    image_list = []
    mask_list = []
    for i, zip_file_path in enumerate(merge_zip_list):
        if zip_file_path.startswith('gs://'):
            local_zip_file_path = 'train_data' + str(i) + '.zip'
            subprocess.check_call([
                'gsutil', '-m', '-q', 'cp', '-r', zip_file_path, local_zip_file_path
            ])
            zip_file_path = local_zip_file_path

        temp_folder_path = 'tmp_folder' + str(i)

        extract_compression_file(zip_file_path, temp_folder_path)

        for root,folder,files in os.walk(temp_folder_path):
            if 'Image' in folder and 'metadata.json' not in os.listdir(root):
                raise ValueError('Please give this [{}] folder "metadata.json" file'.format(root))
            if 'metadata.json' in files:
                src_metadata_list = get_metadata_info(os.path.join(root,'metadata.json'))
                dst_metadata_list,class_num = merge_metadata(src_metadata_list,dst_metadata_list)
                
                write_metadata_info(os.path.join(target_folder_name,'metadata.json'),dst_metadata_list)
            
            if 'Image' in root:
                for image in files:
                    src_image_path = os.path.join(root,image)
                    src_mask_path = os.path.join(root[:-5],'Mask',os.path.splitext(image)[0]+'_mask.png')
                    dst_image_path = os.path.join(target_image_folder,'Image_'+ str(image_cnt) + os.path.splitext(image)[1])
                    dst_mask_path = os.path.join(target_mask_folder,'Image_'+ str(image_cnt) + '.png')                   
                    shutil.copy(src_image_path,dst_image_path)
                    shutil.copy(src_mask_path,dst_mask_path)
                    image_cnt+=1
                    image_list.append(dst_image_path)
                    mask_list.append(dst_mask_path)
        
        shutil.rmtree('tmp_folder' + str(i))

    tf.logging.info("Final metadata {} ".format(dst_metadata_list))
    return (image_list,mask_list),class_num


def get_data(input_path, output_folder):
    if input_path.startswith('Merged;'):
        merge_list = input_path.replace('Merged;', '').split(',')  
    elif input_path.startswith('gs://'):
        subprocess.check_call([
            'gsutil', '-m', '-q', 'cp', '-r', input_path, 'train_data.zip'
        ])
        merge_list = ['train_data.zip']
    else:
        merge_list = input_path.split(',')
        
    data_list,class_num = merger_folder(merge_list, output_folder)
    # Just for Mac user ... delete all the .DS_Store files",
    for arg, dirname, names in os.walk(output_folder):
        if '.DS_Store' in names:
            os.remove(os.path.join(arg, '.DS_Store'))
    return data_list,class_num

# Only for Test
if __name__ == "__main__":
    # merge_list = ['aaa.zip', 'bbb.zip']

    li = 'IRholder.zip,AIRholder.zip'

    get_data(li,'folder_merged')
    # merge_list = li.replace('Merged;', '').split(',')

    # merger_folder(merge_list, 'folder_merged')
