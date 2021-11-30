docker stop gasmeter; docker rm gasmeter;docker build -t gasmeter .;docker run -d --name=gasmeter -it gasmeter python gasmeter.py 2>&1
