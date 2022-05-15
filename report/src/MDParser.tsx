import MarkdownRender from './MarkdownRender';

const TITLE_REGEX = /\w|\s|\//

const MATCHERS = [
    {
        re: /^#{1}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h1 className="w3-wide"><b>{match[1]}</b></h1>
    },
    {
        re: /^#{2}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h2 className="w3-justify"><u>{match[1]}</u></h2>
    }
]

function strToElem(str: string) {
    for (const { re, elem } of MATCHERS) {
        // console.log(str, re.exec(str));
        let match = str.match(re);
        if (match)

            return elem(match);
    }

    return str;
}

var x = 1;

export function ParseMD(mdBody: string): Array<JSX.Element> {
    let mdSplit = mdBody.split("\n");
    // console.log(mdSplit);

    let maped_strs = mdSplit.map(strToElem);

    let elements: Array<JSX.Element> = [];
    let str = "\n";
    maped_strs.forEach((v, i) => {
        console.log(v);
        if (typeof (v) !== "string") {

            if (str !== "\n") {
                elements.push(<div className="w3-justify">
                    <MarkdownRender>{str}</MarkdownRender>
                </div>);
                str = "\n";
            }

            return elements.push(v);
        }

        str += v + "\n";
    })

    if (str !== "\n")
        elements.push(<div className="w3-justify">
            <MarkdownRender>{str}</MarkdownRender>
        </div>);

    return elements;
}