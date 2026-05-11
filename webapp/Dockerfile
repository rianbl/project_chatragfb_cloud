# Step 1: Use a lightweight web server image
FROM nginx:alpine

# Step 2: Set up the working directory
WORKDIR /usr/share/nginx/html

# Step 3: Remove the default NGINX index page
RUN rm -rf ./*

# Step 4: Copy the web app files to the NGINX server directory
COPY index.html styles.css script.js config.json favicon.ico ./

# Step 5: Expose the port
EXPOSE 80

# Step 6: Start the NGINX server
CMD ["nginx", "-g", "daemon off;"]
