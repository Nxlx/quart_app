#!/bin/bash
# you need to give it execute permissions.
# Open the terminal and run the following command:
# chmod +x example.sh
# Now, you can execute the script from the command line using:
#./example.sh
# Set environment variables
export USEAPI_SERVER="1195407304158355486"
export USEAPI_CHANNEL="1196702774650474497"
export USEAPI_TOKEN="user:836-UzQkERrnj2TUn6fFc0H5D"
export USEAPI_DISCORD="MTA4NzAzMTUyNTQ3Nzk5NDUzMA.Gme7e_.DZjv102-bnwa8efcTO9FA_Phfx9KvgnixK7IWI"
export NGROK_AUTHTOKEN="2b2FBcjtykTwVcSY5SxDeCb0oaD_2PCTn5HXdTZDTUq7SGZsd"

# Execute the Python script
python3 ./app.py

