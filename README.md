<h1 style="text-align: center;">
<img src="Assets/logo.png" alt="FaceCraft Logo" width="150" height="150"/>
<br />
🧒 FaceCraft 🧑
</h1>
<p>FaceCraft is a realistic face generator engineered using a StyleGan2 architecture.</p>
<p>FaceCraft's repository lets you train the model yourself or generate a face from a saved checkpoint locally.</p>
<p>Generate a face on our HuggingFace space: <a href="https://huggingface.co/spaces/FaceCraft/FaceCraft" target="_blank">FaceCraft</a></p>

<h2 style="text-align: center;">⚙️ Features ⚙️</h2>
<ul>
<li>High-Resolution Image Generation: Generate images at different resolutions by adjusting the `LOG_RESOLUTION` setting.</li>
<li>Noise Mapping and Style Injection: Implements a noise mapping network and uses style vectors to modulate the weights in the convolution layers.</li>
<li>Gradient Penalty and Path Length Penalty: Includes implementations for WGAN-GP loss and perceptual path length regularization to stabilize training.</li>
<li>Checkpointing: Supports saving and loading model checkpoints to resume training or perform inference at various stages.</li>
</ul>

<h2 style="text-align: center;">🌐 Requirements 🌐</h2>
<p>NVIDIA Drivers</p>
<p>Install python 3.11.9+</p>
<p>Install Docker</p>
<p>Install Flickr-Faces-HQ Dataset: <code>pip install kaggle && kaggle datasets download -d rahulbhalley/ffhq-1024x1024</code></p>
<p>Unzip the images. Images should be organized into subdirectories representing different classes if using `ImageFolder`.</p>
<p>Clone the Repository: <code>git clone https://github.com/EthanStanks/FaceCraft</code></p>

<h2 style="text-align: center;">🔨 Training with Docker 🔨</h2>
<p>Build the docker image (takes 10 minutes)</p>
<p>+ <code>docker build -f Dockerfile_Train -t facecraft-train:1.1 .</code> +</p>
<p>Run the container</p>
<p>+ <code>docker run --gpus all -it -d -e DISPLAY=$DISPLAY -p 6006:6006 -v ${PWD}:/app facecraft-train:1.1</code> +</p>
<p>Open Visual Studio Code</p>
<p>(Optional) Install Python Extension</p>
<p>Attach Current Window to Container</p>
<p>+ <code>Click the "Attach in Current Window" arrow next to "facecraft-train:1.1"</code> +</p>
<p>Navigate to the app directory</p>
<p>+ <code>cd ../app</code> +</p>
<p>Run the Training Script</p>
<p>+ <code>python Training/src/train.py</code> +</p>

<h2 style="text-align: center;">💻 Generating with Docker 💻</h2>
<p>Instructions coming soon</p>

<h2 style="text-align: center;">📸 Dataset 📸</h2>
<p>The Flickr-Faces-HQ Dataset (FFHQ) was used to train and test the GANs and Discriminator models with over 70,000 images of faces. FFHQ Dataset can be found here: <a href="https://github.com/NVlabs/ffhq-dataset" target="_blank">Flickr-Faces-HQ Dataset</a></p>

<h2 style="text-align: center;">👩‍💻 Developers 👨‍💻</h2>
<ul>
<li><a href="https://www.linkedin.com/in/williamhoover70/" target="_blank">Will Hoover</a></li>
<li><a href="https://www.linkedin.com/in/temitayo-shorunke-a520991b5/" target="_blank">Temitayo Shorunke</a></li>
<li><a href="https://www.linkedin.com/in/ethan-stanks/" target="_blank">Ethan Stanks</a></li>
</ul>

<h2 style="text-align: center;">👩‍🏫 Acknowledgment 👨‍🏫</h2>
<p>Special thank you to these individuals who helped us along the way!</p>
<ul>
<li>Rebecca Carroll</li>
<li>Chad Gibson</li>
<li>Philip Smith</li>
</ul>

<h2 style="text-align: center;">⚖️ License ⚖️</h2>
<p>This project is open-sourced under the MIT license. See the LICENSE file for more details.</p>

<h2 style="text-align: center;">✍️ Project Status✍️ </h2>
<p>The FaceCraft team is finished with the project. Check out each of our blogs if you'd like to read about our 5 months of work:</p>
<ul>
<li><a href="https://www.willhoover.dev/" target="_blank">Will's Blog</a></li>
<li><a href="https://analyticalgeniuski.wixsite.com/temitayo-s-portfolio/blog" target="_blank">Temitayo's Blog</a></li>
<li><a href="https://ethanstanks.github.io/capstone_blogs/capstone_blogs.html" target="_blank">Ethan's Blog</a></li>
</ul>
