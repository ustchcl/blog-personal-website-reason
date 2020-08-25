open Utils;
requireCSS("./layout.scss");

[@react.component]
let make = (~style = ReactDOM.Style.make(()), ~className="", ~onClick=(_) => (), ~children) => {
    <div
        onClick={onClick} 
        className={"layout-center" ++ className} 
        style={style}>
        {children}
    </div>
}