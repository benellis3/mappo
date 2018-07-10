echo 'Building Dockerfile with image name pymarl/ngc`'
#docker build --no-cache -t deepmarl/pytorch .
mv .dockerignore .dockerignore.tmp
docker build -t pymarl/ngc -f Dockerfile.ngc .
mv .dockerignore.tmp .dockerignore
