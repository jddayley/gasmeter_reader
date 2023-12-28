publish = False
#publish = True
error_diff = 10
sleep_time = 5
sleep_seconds = 90
train_dataset_path = "data/images_copy"
#train_dataset_path = "data/images"
test_dataset_path = "data/images_test"
camera_dataset_path = "data/camera"
camera_path = "data/camera/output/"
hostIP = "192.168.1.116"
user="admin"
password="2beornot2be"
port=3306
mqqt_q= "gasmeter/reading"
RTSP_HOST="rtsp://jddayley:java@192.168.1.6/live"
mqclientId ="dev_gas_meter_"
area = [350, 400, 1400, 800] # x1, y1, x2, y2
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
   #     print(images.shape)
    mean /= total_images_count
    std /= total_images_count
    return mean, std
