# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from configs.config import get_cfg_defaults
from datasets.creation.util import get_arcface_input, get_center, draw_on
from utils import util
from utils.landmark_detector import LandmarksDetector, detectors

#Sets random seed values for reporducibility
def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False

#Processes images by detecting faces, aligning them, saving data and return paths to output
def process(args, app, image_size=224, draw_bbox=False):
    #If destination directory doesn't exist, then make it
    dst = Path(args.a)
    dst.mkdir(parents=True, exist_ok=True)
    processes = []

    #sort image paths and then iterate through each image
    image_paths = sorted(glob(args.i + '/*.*'))
    for image_path in tqdm(image_paths):
        name = Path(image_path).stem
        img = cv2.imread(image_path)

        #Detect faces from images at paths getting bounding boxes and keypoints
        bboxes, kpss = app.detect(img)
        if bboxes.shape[0] == 0:
            logger.error(f'[ERROR] Face not detected for {image_path}')
            continue
        #If detected, get the face most centered in the image
        i = get_center(bboxes, img)

        #Get bounding box and keypoints of center most detected face
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        
        #Align face using keypoints and bounding box using insightface library
        face = Face(bbox=bbox, kps=kps, det_score=det_score)

        #Prepares image and ArcFace input data
        blob, aimg = get_arcface_input(face, img)

        #Saves processed data for ArcFace as .npy file at path for later
        file = str(Path(dst, name))
        np.save(file, blob)
        processes.append(file + '.npy')

        #Saves the aligned face image as a .jpg file
        cv2.imwrite(file + '.jpg', face_align.norm_crop(img, landmark=face.kps, image_size=image_size))
        if draw_bbox:
            dimg = draw_on(img, [face])
            cv2.imwrite(file + '_bbox.jpg', dimg)

    return processes


def to_batch(path):
    src = path.replace('npy', 'jpg')
    if not os.path.exists(src):
        src = path.replace('npy', 'png')

    image = imread(src)[:, :, :3]
    image = image / 255.
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).cuda()[None]

    arcface = np.load(path)
    arcface = torch.tensor(arcface).cuda()[None]

    return image, arcface


def load_checkpoint(args, mica):
    checkpoint = torch.load(args.m)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])


def main(cfg, args):
    #Set the cuda device and get model to use.
    device = 'cuda:0'
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint(args, mica)
    mica.eval()

    #Retrive face tensor from FLAME model generator
    faces = mica.flameModel.generator.faces_tensor.cpu()
    #Create output directory
    Path(args.o).mkdir(exist_ok=True, parents=True)

    #Use RetinaFace for landmark/face detection 
    app = LandmarksDetector(model=detectors.RETINAFACE)

    with torch.no_grad():
        logger.info(f'Processing has started...')
        #Used to detect and align faces in input images and return paths to processed files
        paths = process(args, app, draw_bbox=False)
        for path in tqdm(paths):
            #Extract name of file path and then load image and ArcFace input data
            name = Path(path).stem
            images, arcface = to_batch(path)

            #Encode the image and ArcFace data to obtain coded representation
            codedict = mica.encode(images, arcface)

            #Decode coded representation to obtain the predictred canonical shap vertices and shape code
            opdict = mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            code = opdict['pred_shape_code']

            #Computes 68 facial landmarks from the predicted mesh vertices using the FLAME model
            lmk = mica.flame.compute_landmarks(meshes)

            #Select 51 relevant landmarks and further reduces it to 7 key landmarks
            mesh = meshes[0]
            landmark_51 = lmk[0, 17:]
            landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

            #Creates directory for each processed image's output
            dst = Path(args.o, name)
            dst.mkdir(parents=True, exist_ok=True)

            #Saves 3D meshes as both .ply and .obj
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
            
            #Saves the shape code, 7 key landmarks, and 68 facial landmarks as .npy files
            np.save(f'{dst}/identity', code[0].cpu().numpy())
            np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
            np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

        logger.info(f'Processing finished. Results has been saved in {args.o}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-i', default='demo/input', type=str, help='Input folder with images')
    parser.add_argument('-o', default='demo/output', type=str, help='Output folder')
    parser.add_argument('-a', default='demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')

    args = parser.parse_args()
    cfg = get_cfg_defaults()

    deterministic(42)
    main(cfg, args)
