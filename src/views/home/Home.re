open Utils;

requireCSS("./home.css");

module Header = {
  let header =
    ReactDOM.Style.make(
      ~height="200px",
      ~marginTop="50px",
      ~width="980px",
      ~display="flex",
      ~flexDirection="column",
      ~justifyContent="space-between",
      (),
    );
  let title =
    ReactDOM.Style.make(
      ~letterSpacing="0.4rem",
      ~color="black",
      ~fontSize="22px",
      ~fontFamily="title-2",
      (),
    );

  let main_title =
    ReactDOM.Style.make(~fontSize="116px", ~fontFamily="title-1", ());
  [@react.component]
  let make = () => {
    <div style=header>
      <div style=title>
        {React.string("TALK IS CHEAP, SHOW ME THE CODE.")}
      </div>
      <div style=main_title> {React.string("Think Different")} </div>
    </div>;
  };
};

module MenuItem = {
  [@react.component]
  let make = (~title, ~router) => {
    <div className="menu-item" onClick={_ => ReasonReactRouter.push(router)}>
      <Center> {React.string(title)} </Center>
    </div>;
  };
};

module Banner = {
  let menus = ["主页", "关于", "我的博客", "联系"];
  [@react.component]
  let make = () => {
    <div className="nav-menu">
      <div className="menu-container">
        <MenuItem title="主页" router="" />
        <MenuItem title="关于" router="/aboutme" />
        <MenuItem title="我的博客" router="/posts" />
        <MenuItem title="联系" router="/contact" />
        <div className="menu-extra">
          <Center> {React.string("github")} </Center>
        </div>
      </div>
    </div>;
  };
};

[@react.component]
let make = () => {
  <div> 
    <Header /> 
    <Banner /> 
    <MarkdownRender source={requireAssetURI("../../assets/md/cnn.md")}/>
    <Footer /> 
  </div>;

};
