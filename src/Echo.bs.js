'use strict';

var React = require("react");

function Echo(Props) {
  var name = Props.name;
  var age = Props.age;
  return React.createElement("div", undefined, "hello, " + (name + (" with " + (age + " old"))));
}

var make = Echo;

exports.make = make;
/* react Not a pure module */
