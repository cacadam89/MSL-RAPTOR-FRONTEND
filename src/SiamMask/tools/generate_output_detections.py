# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *
import os
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('-p', '--prefix_img_name', dest='img_prefix', required=False, default='rgb_',
                    help='Characters before the image ID in the name of the image file')
parser.add_argument('--images_dir', required=True, help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


def generate_label_file(output_dir, bounding_boxes, output_images, images_paths, scores):
    results_path = output_dir+'/results_tracking.txt'
    # Write bounding boxes info
    with open(results_path, 'w') as f:
        for i in range(len(images_paths)):
            print('Writing label '+str(i+1)+' of '+str(len(images_paths)))
            (x3, y3, x4, y4, x1, y1, x2, y2) = bounding_boxes[i]
            f.write(images_paths[i].split('/')[-1].strip('.png')+' '+str(scores[i]) + ' ' +
                    str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')

    # Save images
    for i,im in enumerate(output_images):
        print('Saving img '+str(i+1)+' of '+str(len(output_images)))
        cv2.imwrite(output_dir+'/' + images_paths[i].split('/')[-1], im)




if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    filenames = glob.glob(args.images_dir+"/*.png")
    nb_chars_prefix = len(args.img_prefix)
    filenames_ids = [int(file.split('/')[-1][nb_chars_prefix:-4])
                     for file in filenames]
    filenames_ids.sort()
    images_paths = [args.images_dir+'/'+args.img_prefix +
                    str(filename_id)+'.png' for filename_id in filenames_ids]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI(
            'SiamMask', cv2.imread(images_paths[0]), False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    bounding_boxes = []
    output_images = []
    scores = []
    for f, im_path in enumerate(images_paths):
        im = cv2.imread(im_path)
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            bounding_boxes.append([x,y,x+w,y,x+w,y+w,x,y+w])
            scores.append(1.0)
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            bounding_boxes.append(location)
            scores.append(state['score'])
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            output_images.append(im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    output_dir = args.images_dir+'_tracked_angled_boxes'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_label_file(output_dir, bounding_boxes,
                        output_images, images_paths,scores)
