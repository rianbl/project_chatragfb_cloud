FROM node:20-alpine

WORKDIR /usr/src/app

COPY mcp-server/package.json /usr/src/app/package.json
RUN npm install

COPY mcp-server /usr/src/app
RUN npm run build && npm prune --omit=dev

ENV NODE_ENV=production
ENV MCP_HTTP_PORT=8090

EXPOSE 8090

CMD ["npm", "run", "start"]
