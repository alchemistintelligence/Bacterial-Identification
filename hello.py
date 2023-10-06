import os
from flask import Flask, request,render_template, url_for
from fileinput import filename
import numpy as np
from resnet import ResNet
import torch
from time import time
t00 = time()

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/sukses', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        X_fn = request.files['raman-spectrum_x']
        y_fn = request.files['raman-spectrum_y']
        X = np.load(X_fn, allow_pickle=True)
        y = np.load(y_fn, allow_pickle=True)
        
        # CNN parameters
        layers = 6
        hidden_size = 100
        block_size = 2
        hidden_sizes = [hidden_size] * layers
        num_blocks = [block_size] * layers
        input_dim = 1000
        in_channels = 64
        n_classes = 30
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)
        cuda = torch.cuda.is_available()
        
        # Load trained weights
        cnn = ResNet(hidden_sizes, num_blocks, input_dim,
                    in_channels=in_channels, n_classes=n_classes)

        if cuda: cnn.cuda()
        cnn.load_state_dict(torch.load(
            './finetuned_model.ckpt', map_location=lambda storage, loc: storage))
        
        from training import get_predictions
        from datasets import spectral_dataloader
        
        # Make predictions on subset of data
        t0 = time()
        dl = spectral_dataloader(X, y, batch_size=10, shuffle=False)
        y_hat = get_predictions(cnn, dl, cuda)
        # print('Predicted {} spectra: {:0.2f}s'.format(len(y_hat), time()-t0))
        
        # Computing accuracy
        acc = (y_hat == y).mean()
        # print('Accuracy: {:0.1f}%'.format(100*acc))
        
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        from config import ORDER, STRAINS
        
        # Plot confusion matrix
        sns.set_context("talk", rc={"font":"Helvetica", "font.size":12})
        label = [STRAINS[i] for i in ORDER]
        cm = confusion_matrix(y, y_hat, labels=ORDER)
        plt.figure(figsize=(15, 12))
        cm = 100 * cm / cm.sum(axis=1)[:,np.newaxis]
        ax = sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt='0.0f',
                        xticklabels=label, yticklabels=label)
        ax.xaxis.tick_top()
        plt.xticks(rotation=90) 
        plt.savefig('./static/pyplot.png')
        
        
        return render_template('result.html')