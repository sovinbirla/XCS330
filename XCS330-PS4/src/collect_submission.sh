rm -f ps4.zip 
pushd submission; zip -r ../ps4.zip . --exclude "__pycache__/*"; popd