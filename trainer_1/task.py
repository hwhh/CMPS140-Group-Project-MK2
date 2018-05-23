
def load_melgram(file_path):
    #auto-detect load method based on filename extension
    name, extension = os.path.splitext(file_path)
    if ('.npy' == extension):
        melgram = np.load(file_path)
    elif ('.npz' == extension):          # compressed npz file (preferred)
        with np.load(file_path) as data:
            melgram = data['melgram']
    elif ('.png' == extension) or ('.jpeg' == extension):
        arr = imread(file_path)
        melgram = np.reshape(arr, (1,1,arr.shape[0],arr.shape[1]))  # convert 2-d image
        melgram = np.flip(melgram, 0)     # we save images 'rightside up' but librosa internally presents them 'upside down'
    else:
        print("load_melgram: Error: unrecognized file extension '",extension,"' for file ",file_path,sep="")
    return melgram

def shuffle_XY_paths(X,Y,paths):   # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0] )
    #print("shuffle_XY_paths: Y.shape[0], len(paths) = ",Y.shape[0], len(paths))
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths[:]
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths

def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None

def build_data_set(path, load_frac=1.0, batch_size=None, tile=False):
    test = pd.read_csv("../input/sample_submission.csv")
    train = pd.read_csv("../input/train.csv")

    labels = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(labels)}
    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

    X = np.empty(shape=(config.nb_classes, config.shape[1], config.dim[2], config.dim[3]))
    paths = []

    load_count = 0
    for idx, label in enumerate(LABELS):
        this_Y = np.array(encode_class(label, labels))
        this_Y = this_Y[np.newaxis, :]
        class_files = train['label_idx'][label]#Get the files for specific class
        n_files = len(class_files)
        n_load = int(n_files * load_frac)  # n_load is how many files of THIS CLASS are expected to be loaded
        file_list = class_files[0:n_load]
        for idx2, infilename in enumerate(file_list):  # Load files in a particular class
            audio_path = path + classname + '/' + infilename
            melgram = load_melgram(audio_path)
            X[load_count, :, :] = melgram
            Y[load_count, :] = this_Y
            paths.append(audio_path)
            load_count += 1
            if (load_count >= total_load):  # Abort loading files after last even multiple of batch size
                break
        if (load_count >= total_load):  # Second break needed to get out of loop over classes
            break
    if (load_count != total_load):  # check to make sure we loaded everything we thought we would
        raise Exception("Loaded " + str(load_count) + " files but was expecting " + str(total_load))

    X, Y, paths = shuffle_XY_paths(X, Y, paths)  # mix up classes, & files within classes

    return X, Y, paths, class_names
