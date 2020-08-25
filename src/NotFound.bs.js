'use strict';

var React = require("react");
var ReasonReactRouter = require("reason-react/src/ReasonReactRouter.bs.js");

var not_found_container = {
  display: "flex",
  margin: "30px auto",
  alignItems: "center",
  flexDirection: "column"
};

var not_found_image = {
  marginTop: "60px"
};

var not_found_text = {
  marginTop: "40px"
};

function NotFound(Props) {
  return React.createElement("div", {
              style: not_found_container
            }, React.createElement("button", {
                  onClick: (function (param) {
                      return ReasonReactRouter.push("/hello/rust");
                    })
                }, "Goto Echo"), React.createElement("div", {
                  style: not_found_image
                }), React.createElement("div", {
                  style: not_found_text
                }, React.createElement("span", undefined, "The page you're looking for can't be found. Go home by ")));
}

var make = NotFound;

exports.not_found_container = not_found_container;
exports.not_found_image = not_found_image;
exports.not_found_text = not_found_text;
exports.make = make;
/* react Not a pure module */
