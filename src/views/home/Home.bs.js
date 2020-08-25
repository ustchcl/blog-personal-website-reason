'use strict';

var React = require("react");
var ReasonReactRouter = require("reason-react/src/ReasonReactRouter.bs.js");
var Center$BlogReasonreact = require("../../Components/Layout/Center.bs.js");
var Footer$BlogReasonreact = require("./Footer.bs.js");
var MarkdownRender$BlogReasonreact = require("../../Components/MarkdownRender.bs.js");

require("./home.css");

var header = {
  display: "flex",
  height: "200px",
  marginTop: "50px",
  width: "980px",
  flexDirection: "column",
  justifyContent: "space-between"
};

var title = {
  color: "black",
  fontFamily: "title-2",
  fontSize: "22px",
  letterSpacing: "0.4rem"
};

var main_title = {
  fontFamily: "title-1",
  fontSize: "116px"
};

function Home$Header(Props) {
  return React.createElement("div", {
              style: header
            }, React.createElement("div", {
                  style: title
                }, "TALK IS CHEAP, SHOW ME THE CODE."), React.createElement("div", {
                  style: main_title
                }, "Think Different"));
}

var Header = {
  header: header,
  title: title,
  main_title: main_title,
  make: Home$Header
};

function Home$MenuItem(Props) {
  var title = Props.title;
  var router = Props.router;
  return React.createElement("div", {
              className: "menu-item",
              onClick: (function (param) {
                  return ReasonReactRouter.push(router);
                })
            }, React.createElement(Center$BlogReasonreact.make, {
                  children: title
                }));
}

var MenuItem = {
  make: Home$MenuItem
};

function Home$Banner(Props) {
  return React.createElement("div", {
              className: "nav-menu"
            }, React.createElement("div", {
                  className: "menu-container"
                }, React.createElement(Home$MenuItem, {
                      title: "\xe4\xb8\xbb\xe9\xa1\xb5",
                      router: ""
                    }), React.createElement(Home$MenuItem, {
                      title: "\xe5\x85\xb3\xe4\xba\x8e",
                      router: "/aboutme"
                    }), React.createElement(Home$MenuItem, {
                      title: "\xe6\x88\x91\xe7\x9a\x84\xe5\x8d\x9a\xe5\xae\xa2",
                      router: "/posts"
                    }), React.createElement(Home$MenuItem, {
                      title: "\xe8\x81\x94\xe7\xb3\xbb",
                      router: "/contact"
                    }), React.createElement("div", {
                      className: "menu-extra"
                    }, React.createElement(Center$BlogReasonreact.make, {
                          children: "github"
                        }))));
}

var Banner_menus = {
  hd: "\xe4\xb8\xbb\xe9\xa1\xb5",
  tl: {
    hd: "\xe5\x85\xb3\xe4\xba\x8e",
    tl: {
      hd: "\xe6\x88\x91\xe7\x9a\x84\xe5\x8d\x9a\xe5\xae\xa2",
      tl: {
        hd: "\xe8\x81\x94\xe7\xb3\xbb",
        tl: /* [] */0
      }
    }
  }
};

var Banner = {
  menus: Banner_menus,
  make: Home$Banner
};

function Home(Props) {
  return React.createElement("div", undefined, React.createElement(Home$Header, {}), React.createElement(Home$Banner, {}), React.createElement(MarkdownRender$BlogReasonreact.make, {
                  source: require("../../assets/md/cnn.md")
                }), React.createElement(Footer$BlogReasonreact.make, {}));
}

var make = Home;

exports.Header = Header;
exports.MenuItem = MenuItem;
exports.Banner = Banner;
exports.make = make;
/*  Not a pure module */
