from interfaces.http import create_app

app = create_app()


if __name__ == "__main__":
    container = app.config["service_container"]
    container.startup_service.run()
    app.run(host="0.0.0.0", port=8080)

