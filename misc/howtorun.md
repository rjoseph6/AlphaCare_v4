# How To Run 
1. Open the terminal and run the following command: 

![](ui/terminal.png)

```bash
cd Downloads 
```
```bash
git clone https://github.com/rjoseph6/AlphaCare_v4.git
```
We are downloading the project from the repository.

MAC
```
curl -o test_img.jpg "https://uvderm.com/wp-content/uploads/2023/03/Actinic-Keratosis.jpeg"
```
WINDOWS
```
wget "https://uvderm.com/wp-content/uploads/2023/03/Actinic-Keratosis.jpeg" -O test_img.jpg
```

2. Run these commands to start the frontend:

```bash
cd AlphaCare_v4/frontend
```

```bash
npm install
```

```bash
npm start
```

4. A browser window should open with the frontend running. 

![ui](ui/v1_frontend2.png)

5. Start a NEW second terminal

6. Run these commands in the new terminal

```bash
cd Downloads/AlphaCare_v4/backend 
```
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```
```bash
pip3 install -r requirements.txt
```
```bash
python3 app.py
```

6. The frontend (browser) should now be connected to the backend (python script). 

7. Go to the browser that was initially opened. Click on "Choose File". In Downloads Folder you should see:
```
test_img.jpg
```
![](ui/downloads.png)

8. Click the "Submit" button. The model will predict the image and display the results.

![ui](ui/v1_frontend.png)

9. More detailed output of the model can be seen in the terminal where the backend is running.

![](ui/backend_terminal.png)