'use strict';

var React = require("react");
var Center$BlogReasonreact = require("../../Components/Layout/Center.bs.js");

function Footer(Props) {
  var foot = {
    background: "#e8e6e6",
    color: "black",
    fontSize: "14px",
    height: "46px",
    width: "100%"
  };
  return React.createElement("div", {
              style: foot
            }, React.createElement(Center$BlogReasonreact.make, {
                  children: "\xc2\xa92020 by HongShen (ustchcl@gmail.com)"
                }));
}

var make = Footer;

exports.make = make;
/* react Not a pure module */
