// open Utils;

let not_found_container = ReactDOM.Style.make
    ( ~margin = "30px auto"
    , ~display = "flex"
    , ~alignItems = "center"
    , ~flexDirection = "column"
    , ());

let not_found_image = ReactDOM.Style.make
    ( ~marginTop="60px"
    , ()
    );

let not_found_text = ReactDOM.Style.make
    ( ~marginTop="40px"
    , ()
    );

// let notFoundImage = requireAssetURI("src/assets/image/notfound404.png");

[@react.component]
let make = () =>
  <div style={not_found_container}>
    <button onClick={_ => ReasonReactRouter.push("/hello/rust")}>{React.string("Goto Echo")}</button>
    <div style={not_found_image}>
      // <img alt="Page not found" src=notFoundImage />
    </div>
    <div style={not_found_text}> 
      <span>
        {React.string(
           "The page you're looking for can't be found. Go home by ",
         )}
      </span>
    </div>
  </div>;
