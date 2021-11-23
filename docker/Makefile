
help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose exec handson-ml3 bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --no-cache
stop:
	docker stop handson-ml3 || true; docker rm handson-ml3 || true;
.FORCE:
