# How to use the Docker Dev Environment:

## Setup Docker Dev Environment using Docker Desktop

1. Open Docker Desktop
1. Go to Dev Environments
1. Click Create
1. Click Get Start
1. Give a name and copy paste your github address
1. Click Continue
1. You are all set

## Setup Docker Dev Environment by just one command

`docker dev create --name mlenv -d https://github.com/qiaoxingit/ML.git`

## Create Docker Dev Environment

This command will create the docker image that can be used as Dev Environment:

`docker build -t denglufei/ml_2023_spring_env:1.0 .`
