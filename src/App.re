[@react.component]
let make = () => {
    let url = ReasonReact.Router.useUrl();
    Js.Console.log("==== path ====");
    Js.Console.log(url.path);

    switch (url.path) {
    | [] => <Home/>
    | [name, age] => <Echo name={name} age={age}/>
    | _ => <NotFound/>
    }
}