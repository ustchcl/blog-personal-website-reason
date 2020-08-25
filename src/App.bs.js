'use strict';

var React = require("react");
var ReasonReactRouter = require("reason-react/src/ReasonReactRouter.bs.js");
var Echo$BlogReasonreact = require("./Echo.bs.js");
var Home$BlogReasonreact = require("./views/home/Home.bs.js");
var NotFound$BlogReasonreact = require("./NotFound.bs.js");

function App(Props) {
  var url = ReasonReactRouter.useUrl(undefined, undefined);
  console.log("==== path ====");
  console.log(url.path);
  var match = url.path;
  if (!match) {
    return React.createElement(Home$BlogReasonreact.make, {});
  }
  var match$1 = match.tl;
  if (match$1) {
    if (match$1.tl) {
      return React.createElement(NotFound$BlogReasonreact.make, {});
    } else {
      return React.createElement(Echo$BlogReasonreact.make, {
                  name: match.hd,
                  age: match$1.hd
                });
    }
  } else {
    return React.createElement(NotFound$BlogReasonreact.make, {});
  }
}

var make = App;

exports.make = make;
/* react Not a pure module */
