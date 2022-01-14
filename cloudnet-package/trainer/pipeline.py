import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

IMG_HEIGHT = tf.constant(192, dtype=tf.int32)
IMG_WIDTH = tf.constant(192,dtype=tf.int32)
EPOCHS  = tf.constant(2,dtype=tf.int32)

@tf.function
def load_image(img_path):
    image = tf.io.read_file(img_path)
    image = tfio.experimental.image.decode_tiff(image)
    
    return image

@tf.function
def parser(row_csv):
    # decoding the csv file
    chip_id,B02_path,B03_path,B04_path,B08_path,label_path = tf.io.decode_csv(records = row_csv, record_defaults = ["chip_id","B02_path","B03_path","B04_path","B08_path","label_path"], field_delim=';')
    
    B02_img = load_image(B02_path)
    # B02_img = B02_img[:,:,0]
    # B02_img = tf.expand_dims(B02_img, axis = -1, name=None)
    
    B03_img = load_image(B03_path)
    # B03_img = B03_img[:,:,0]
    # B03_img = tf.expand_dims(B03_img, axis = -1, name=None)
    
    B04_img = load_image(B04_path)
    # B04_img = B04_img[:,:,0]
    # B04_img = tf.expand_dims(B04_img, axis = -1, name=None)
    
    B08_img = load_image(B08_path)
    # B08_img = B08_img[:,:,0]
    # B08_img = tf.expand_dims(B08_img, axis = -1, name=None)
    
    image = tf.concat([B02_img,B03_img,B04_img,B08_img], axis = -1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)
    
    label = load_image(label_path)
    # label = label[:,:,0]
    # label = tf.expand_dims(label, axis = -1, name=None)
    
    label = tf.image.convert_image_dtype(label, dtype=tf.float32, saturate=False)
    
    
    return image, label

@tf.function
def img_reshape(img,msk):
    img_o = tf.image.resize(img,(192, 192),preserve_aspect_ratio=False)
    msk_o = tf.image.resize(msk,(192, 192),preserve_aspect_ratio=False)
    
    return img_o,msk_o
    

@tf.function
def rotate_clk_img_and_msk(img, msk):
    angles_tensor = tf.constant([4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=tf.float32)
    angle = tf.random.shuffle(angles_tensor)[0]
    # Image
    img_o = tfa.image.rotate(images = img,angles = angle,interpolation = "nearest",fill_mode = "reflect",fill_value = 0.0)
    # Label
    msk_o = tfa.image.rotate(images = msk,angles = angle,interpolation = "nearest",fill_mode = "reflect",fill_value = 0.0)
    
    return img_o, msk_o

@tf.function
def rotate_cclk_img_and_msk(img, msk):
    angles_tensor = tf.constant([-20, -18, -16, -14, -12, -10, -8, -6, -4],dtype=tf.float32)
    angle = tf.random.shuffle(angles_tensor)[0]
    # Image
    img_o = tfa.image.rotate(images = img,angles = angle,interpolation = "nearest",fill_mode = "reflect",fill_value = 0.0)
    # Label
    msk_o = tfa.image.rotate(images = msk,angles = angle,interpolation = "nearest",fill_mode = "reflect",fill_value = 0.0)
    
    return img_o, msk_o

@tf.function
def flipping_img_and_msk(img, msk):
    img_o = tf.image.flip_left_right(img)
    img_o = tf.image.flip_up_down(img)
    
    msk_o = tf.image.flip_left_right(msk)
    msk_o = tf.image.flip_up_down(msk)
    
    return img_o,msk_o


    
@tf.function
def zoom_img_and_msk(img, msk,height = 512,width = 512):

    zoom_factor_tensor = tf.constant([1.2, 1.5, 1.8, 2, 2.2, 2.5], dtype=tf.float32)  # currently doesn't have zoom out!
    zoom_factor = tf.random.shuffle(zoom_factor_tensor)[0]
    # print("*-*-*-*-*-*")
    # print(img.shape)
    #h,w,c = img.shape
    # h = height
    # w = width
    h = tf.cast(height,dtype= tf.int32)
    w = tf.cast(width,dtype= tf.int32)
    
    # img = tf.cast(img, dtype=tf.float32)
    # msk = tf.cast(msk, dtype=tf.float32)
    

    # width and height of the zoomed image
    zh = tf.math.multiply(zoom_factor, tf.cast(h, dtype=tf.float32))
    zh = tf.cast(zh,dtype= tf.int32)
    
    zw = tf.math.multiply(zoom_factor, tf.cast(w, dtype=tf.float32))
    zw = tf.cast(zw,dtype= tf.int32)
    
    # zh = int(np.round(zoom_factor * h))
    # zw = int(np.round(zoom_factor * w))

    img = tf.image.resize(img,(zh, zw),preserve_aspect_ratio=False)
    msk = tf.image.resize(msk,(zh, zw),preserve_aspect_ratio=False)
    
    region_tensor = tf.constant([0, 1, 2, 3, 4],dtype= tf.float32)
    region = tf.random.shuffle(region_tensor)[0]

    # zooming out
    # tf.print("before zoom")
    # tf.print(img.dtype)
    
    if tf.math.less_equal(zoom_factor, tf.constant(1, dtype= tf.float32)):
        outimg = img
        outmsk = msk
        
        # tf.print("zoom 1")
        # tf.print(img.dtype)

    # zooming in
    # else:
    #     # Initializing
    #     outimg = tf.zeros_like(img, dtype=tf.float32, name=None)
    #     outmsk = tf.zeros_like(msk, dtype=tf.float32, name=None)
        # bounding box of the clipped region within the input array
    elif tf.math.equal(region, tf.constant(0,dtype= tf.float32)):
        outimg = img[0:h, 0:w,:]
        outmsk = msk[0:h, 0:w,:]
        
        # tf.print("zoom 0")
        # tf.print(img.dtype)
        
    elif tf.math.equal(region, tf.constant(1,dtype= tf.float32)):
        outimg = img[0:h, zw - w:zw,:]
        outmsk = msk[0:h, zw - w:zw,:]
        
        # tf.print("zoom 11")
        # tf.print(img.dtype)
        
    elif tf.math.equal(region, tf.constant(2,dtype= tf.float32)):
        outimg = img[zh - h:zh, 0:w,:]
        outmsk = msk[zh - h:zh, 0:w,:]
        
        # tf.print("zoom 2")
        # tf.print(img.dtype)
        
    elif tf.math.equal(region, tf.constant(3,dtype= tf.float32)):
        outimg = img[zh - h:zh, zw - w:zw,:]
        outmsk = msk[zh - h:zh, zw - w:zw,:]
        
        # tf.print("zoom 3")
        # tf.print(img.dtype)
        
    # if tf.math.equal(region, tf.constant(4,dtype= tf.float32)):

    else:
        # tf.print("zoom 4")
        # tf.print(img.dtype)
        
        marh = tf.math.floordiv( h, tf.constant(2))
        marw = tf.math.floordiv( w, tf.constant(2))

        zh_div = tf.math.floordiv( zh, tf.constant(2))
        zw_div = tf.math.floordiv( zw, tf.constant(2))

        zh_div_add = tf.math.add( zh_div, marh)
        zh_div_minus = tf.math.subtract( zh_div, marh)

        zw_div_add = tf.math.add( zw_div, marw)
        zw_div_minus = tf.math.subtract( zw_div, marw)

        outimg = img[zh_div_minus:zw_div_add, zw_div_minus:zw_div_add,:]
        outmsk = msk[zh_div_minus:zw_div_add, zw_div_minus:zw_div_add,:]

        # outimg = img[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw),:]
        # outmsk = msk[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw),:]

    # to make sure the output is in the same size of the input
    img_o = tf.image.resize(outimg,(h, w),preserve_aspect_ratio=False)
    msk_o = tf.image.resize(outmsk,(h, w),preserve_aspect_ratio=False)
    return img_o, msk_o


@tf.function
def data_augmentation(img, msk):
    
    coin_rotate_clk_img_and_msk = tf.random.uniform(shape = (1,1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
    coin_rotate_cclk_img_and_msk = tf.random.uniform(shape = (1,1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
    coin_flipping_img_and_msk = tf.random.uniform(shape = (1,1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
    coin_zoom_img_and_msk = tf.random.uniform(shape = (1,1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)

    # rotate_clk_img_and_msk
    if tf.math.greater_equal(coin_rotate_clk_img_and_msk, tf.constant(0.5)):
        # tf.print(tf.constant("rotate_clk_img_and_msk"))
        # tf.print(coin_rotate_clk_img_and_msk)
        img,msk = rotate_clk_img_and_msk(img, msk)
    
    # rotate_cclk_img_and_msk
    if tf.math.greater_equal(coin_rotate_cclk_img_and_msk, tf.constant(0.5)):
        # tf.print(tf.constant("rotate_cclk_img_and_msk"))
        # tf.print(coin_rotate_cclk_img_and_msk)
        img,msk = rotate_cclk_img_and_msk(img, msk)
        
    # flipping_img_and_msk
    # rotate_cclk_img_and_msk
    if tf.math.greater_equal(coin_flipping_img_and_msk, tf.constant(0.5)):
        # tf.print(tf.constant("flipping_img_and_msk"))
        # tf.print(coin_flipping_img_and_msk)
        img,msk = flipping_img_and_msk(img, msk)
    
    # zoom_img_and_msk
    if tf.math.greater_equal(coin_zoom_img_and_msk, tf.constant(0.5)):
        # tf.print(tf.constant("coin_zoom_img_and_msk"))
        # tf.print(coin_zoom_img_and_msk)
        img,msk = zoom_img_and_msk(img, msk,height = IMG_HEIGHT,width = IMG_WIDTH)
    
    return img,msk

def load_dataset(file_paths, reshape = False,buffer_size = 1000, batch_size = 12,training = True, num_epochs = EPOCHS):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    dataset = tf.data.TextLineDataset(filenames=file_paths)
    dataset = dataset.skip(1).map(parser).cache()
    
    # reshape
    if reshape == True:
        dataset = dataset.map(img_reshape)
    
    dataset = dataset.map(data_augmentation)
    
    if training == True:
        dataset = dataset.shuffle(buffer_size =buffer_size)
        dataset = dataset.repeat(count=num_epochs)
    else:
        dataset = dataset.repeat(count=1)
        
    dataset = dataset.batch(batch_size = batch_size,drop_remainder = False).prefetch(buffer_size=AUTOTUNE)
    
    return dataset
    

