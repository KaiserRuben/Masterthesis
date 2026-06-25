/** @type {import('next').NextConfig} */
const path = require("path");

const appDir = __dirname;

const nextConfig = {
  output: "standalone",
  env: {
    HS01_POOL:
      process.env.HS01_POOL ||
      path.resolve(appDir, "../pool_frozen/itempool.json"),
    HS01_CONFIG:
      process.env.HS01_CONFIG ||
      path.resolve(appDir, "./config/study-config.json"),
    HS01_DATA_DIR:
      process.env.HS01_DATA_DIR || path.resolve(appDir, "./data/sessions"),
    HS01_IMAGE_DIR:
      process.env.HS01_IMAGE_DIR ||
      path.resolve(appDir, "../pool_frozen/assets/images"),
    HS01_REFERENCES:
      process.env.HS01_REFERENCES ||
      path.resolve(appDir, "./config/references.json"),
    HS01_REFS_DIR:
      process.env.HS01_REFS_DIR || path.resolve(appDir, "./config/refs"),
  },
};

module.exports = nextConfig;
