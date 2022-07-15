import os
import tensorflow as tf
import config as config
import numpy as np
import cv2
import dataClass as data
from keras import applications
from keras.models import load_model


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def reconstruct(batchX, predictedY, filelist):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_results_path = os.path.join(config.OUT_DIR,config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist +  "_reconstructed.jpg" )
    cv2.imwrite(save_path, result)
    return result

def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def sample_images():
    avg_cost = 0
    avg_cost2 = 0
    avg_cost3 = 0
    avg_ssim = 0
    avg_psnr = 0
    VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)
    save_models_path =os.path.join(config.MODEL_DIR,config.TEST_NAME)
    save_path = os.path.join(config.MODEL_DIR, config.PRETRAINED)
    print(save_path)
    colorizationModel = load_model(save_path)
    test_data = data.DATA(config.TEST_DIR)
    assert config.BATCH_SIZE<=test_data.size, "The batch size should be smaller or equal to the number of testing images --> modify it in config.py"
    total_batch = int(test_data.size/config.BATCH_SIZE)
    print("number of images to inpaint " + str(test_data.size))
    print("total number of batches to colorize " + str(total_batch))
    for b in range(total_batch):
            #batchX, batchY,  filelist  = test_data.generate_batch()
            batchX, batchY,  filelist, original, labimg_oritList = test_data.generate_batch()
            predY, _  = colorizationModel.predict(np.tile(batchX,[1,1,1,3]))
            predictVGG =VGG_modelF.predict(np.tile(batchX,[1,1,1,3]))
            loss = colorizationModel.evaluate(np.tile(batchX,[1,1,1,3]), [batchY, predictVGG], verbose=0)
            avg_cost += loss[0]
            avg_cost2 += loss[1]
            avg_cost3 += loss[2]
            for i in range(config.BATCH_SIZE):
                originalResult_red = reconstruct_no(deprocess(batchX)[i], deprocess(batchY)[i])
                predResult_red = reconstruct_no(deprocess(batchX)[i], deprocess(predY)[i])
                ssim= tf.keras.backend.eval( tf.image.ssim(tf.convert_to_tensor(originalResult_red, dtype=tf.float32), tf.convert_to_tensor(predResult_red, dtype=tf.float32), max_val=255))
                psnr= tf.keras.backend.eval( tf.image.psnr(tf.convert_to_tensor(originalResult_red, dtype=tf.float32), tf.convert_to_tensor(predResult_red, dtype=tf.float32), max_val=255))
                avg_ssim += ssim
                avg_psnr += psnr

                originalResult = original[i]
                height, width, channels = originalResult.shape
                predictedAB = cv2.resize(deprocess(predY[i]), (width,height))
                labimg_ori =np.expand_dims(labimg_oritList[i],axis=2)
                predResult= reconstruct_no(deprocess(labimg_ori), predictedAB)
                save_path = os.path.join(config.OUT_DIR, "{:4.8f}_".format(psnr)+filelist[i][:-4] +"psnr_reconstructed.jpg" )
                cv2.imwrite(save_path, np.concatenate((predResult, originalResult)))
                print("Batch " + str(b)+"/"+str(total_batch))
                print(psnr)

    print(" ----------  loss =", "{:.8f}------------------".format(avg_cost/total_batch))
    print(" ----------  upsamplingloss =", "{:.8f}------------------".format(avg_cost2/total_batch))
    print(" ----------  classification_loss =", "{:.8f}------------------".format(avg_cost3/total_batch))
    print(" ----------  ssim loss =", "{:.8f}------------------".format(avg_ssim/(total_batch*config.BATCH_SIZE)))
    print(" ----------  psnr loss =", "{:.8f}------------------".format(avg_psnr/(total_batch*config.BATCH_SIZE)))



if __name__ == '__main__':
    sample_images()
