'use strict';

var React = require("react");
var Caml_option = require("bs-platform/lib/js/caml_option.js");

require("./layout.scss");

function Center(Props) {
  var styleOpt = Props.style;
  var classNameOpt = Props.className;
  var onClickOpt = Props.onClick;
  var children = Props.children;
  var style = styleOpt !== undefined ? Caml_option.valFromOption(styleOpt) : ({});
  var className = classNameOpt !== undefined ? classNameOpt : "";
  var onClick = onClickOpt !== undefined ? onClickOpt : (function (param) {
        
      });
  return React.createElement("div", {
              className: "layout-center" + className,
              style: style,
              onClick: onClick
            }, children);
}

var make = Center;

exports.make = make;
/*  Not a pure module */
