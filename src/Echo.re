[@react.component]
let make = (~name, ~age) => {
    <div>
        {React.string("hello, " ++ name ++ " with " ++ age ++ " old")}
    </div>
}