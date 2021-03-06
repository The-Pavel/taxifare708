# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* taxifare/*.py

black:
	@black scripts/* taxifare/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr taxifare-*.dist-info
	@rm -fr taxifare.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

JOB_NAME='taxifare_linreg_training_$(shell date +'%Y%m%d_%H%M%S')'
BUCKET_NAME='lewagon-data-708-pavel'
BUCKET_TRAINING_FOLDER='trainings'
PACKAGE_NAME='taxifare'
FILENAME='trainer'
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15
REGION='us-west1'

submit_training:
	@gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

PROJECT_ID='majestic-voice-331803'
DOCKER_IMAGE_NAME='taxifare-api'
REGISTRY_REGION='us.gcr.io'

build_gcloud_image:
	@docker build -t ${REGISTRY_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

push_gcr_image:
	@docker push ${REGISTRY_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

test_gcr_image_locally:
	@docker run -e PORT=8000 -p 8080:8000 ${REGISTRY_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

deploy_gcr_container:
	@gcloud run deploy --image ${REGISTRY_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region ${REGION}
