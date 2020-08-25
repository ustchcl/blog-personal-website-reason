'use strict';

var React = require("react");
var ReactDOMRe = require("reason-react/src/legacy/ReactDOMRe.bs.js");
var ReasonReactRouter = require("reason-react/src/ReasonReactRouter.bs.js");
var App$BlogReasonreact = require("./App.bs.js");
var Utils$BlogReasonreact = require("./Utils.bs.js");
var ExampleStyles$BlogReasonreact = require("./ExampleStyles.bs.js");

require("./index.css");

var style = document.createElement("style");

document.head.appendChild(style);

style.innerHTML = ExampleStyles$BlogReasonreact.style;

Utils$BlogReasonreact.unregister(undefined);

ReactDOMRe.renderToElementWithId(React.createElement(App$BlogReasonreact.make, {}), "app");

ReasonReactRouter.push("");

exports.style = style;
/*  Not a pure module */
