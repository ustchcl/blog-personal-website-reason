'use strict';

var ServiceWorkerJs = require("./serviceWorker.js");

function fromNow(unixtime) {
  var delta = (Date.now() / 1000 | 0) - unixtime | 0;
  if (delta < 3600) {
    return String(delta / 60 | 0) + " minutes ago";
  } else if (delta < 86400) {
    return String(delta / 3600 | 0) + " hours ago";
  } else {
    return String(delta / 86400 | 0) + " days ago";
  }
}

function dangerousHtml(html) {
  return {
          __html: html
        };
}

function distanceFromBottom(param) {
  var bodyClientHeight = document.body.clientHeight;
  var windowScrollY = window.scrollY;
  var windowInnerHeight = window.innerHeight;
  return bodyClientHeight - (windowScrollY + windowInnerHeight | 0) | 0;
}

function register(prim) {
  ServiceWorkerJs.register();
  
}

function unregister(prim) {
  ServiceWorkerJs.unregister();
  
}

exports.fromNow = fromNow;
exports.dangerousHtml = dangerousHtml;
exports.distanceFromBottom = distanceFromBottom;
exports.register = register;
exports.unregister = unregister;
/* ./serviceWorker.js Not a pure module */
