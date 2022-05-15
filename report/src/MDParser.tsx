import MarkdownRender from './MarkdownRender';


const MATCHES = [
    { re: /^#{1}\s.*/, elem: (str: string) => <h1 className="w3-wide">{str}</h1> }
]

function strToElem(str: string) {
    for (const { re, elem } of MATCHES) {
        console.log(str, re.exec(str));
        if (re.exec(str))
            re.split();

        return elem(str);
    }

    return str;
}

var x = 1;

export function ParseMD(mdBody: string): Array<JSX.Element> {
    let mdSplit = mdBody.split("\n");
    // console.log(mdSplit);

    let maped_strs = mdSplit.map(strToElem);

    let elements: Array<JSX.Element> = [];
    let str = "";
    maped_strs.forEach((v, i) => {
        if (typeof (v) !== "string") {

            if (str !== "") {
                elements.push(<MarkdownRender>{str}</MarkdownRender>);
                str = "";
            }

            return elements.push(v);
        }

        str += v;
    })

    return elements;
}