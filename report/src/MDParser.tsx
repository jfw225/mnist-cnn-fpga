import MarkdownRender from './MarkdownRender';

/** The regex used to match container comments. */
const CONTAINER_RE = /^<!-- container:\s*(\w*)\s*-->/;

/** Object used to match strings to their proper formatters. */
const MATCHERS = [
    {
        re: /^#{1}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h1 className="w3-wide"><b>{match[1]}</b></h1>
    },
    {
        re: /^#{2}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h2 className="w3-justify"><u>{match[1]}</u></h2>
    },
    {
        re: /^#{4}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h4 className="w3-justify"><u>{match[1]}</u></h4>
    }
]

type ContainerElement = JSX.Element | Array<JSX.Element>;

/** The container keys. */
enum CONTAINER_KEY {
    DEFAULT = "default",
    DARK = "dark"
};


/** Dictionary of container creators. */
const CONTAINER_FNS = {
    [CONTAINER_KEY.DEFAULT]: (e: ContainerElement) => (
        <div className="w3-white">
            <div className="w3-container w3-content w3-center w3-padding-64" style={{ maxWidth: 800 }} id="report">
                {e}
            </div>
        </div>
    ),
    [CONTAINER_KEY.DARK]: (e: ContainerElement) => (
        <div className="w3-black">
            <div className="w3-container w3-content w3-center w3-padding-64" style={{ maxWidth: 800 }} id="report">
                {e}
            </div>
        </div>
    )
}

function createContainer(key: CONTAINER_KEY, e: ContainerElement): JSX.Element {
    if (!key || !CONTAINER_FNS.hasOwnProperty(key))
        key = CONTAINER_KEY.DEFAULT;

    return CONTAINER_FNS[key](e);
}

function strToElem(str: string) {
    for (const { re, elem } of MATCHERS) {
        // console.log(str, re.exec(str));
        let match = str.match(re);
        if (match)

            return elem(match);
    }

    return str;
}



export function ParseMD(mdBody: string): ContainerElement {
    let mdSplit = mdBody.split("\n");
    // console.log(mdSplit);

    /* The key used to find the container function. */
    let containerKey: CONTAINER_KEY = CONTAINER_KEY.DEFAULT;

    let containers: Array<JSX.Element> = [];

    /* Array to keep track of elements to be placed into a container. */
    let elements: Array<JSX.Element> = [];

    /* String to keep track of overflow. */
    let str = "\n";

    /* Function used to convert `str` to JSX element and add to `elements`. */
    const handleStr = () => {
        if (str !== "\n") {
            elements.push(
                <div className="w3-justify">
                    <MarkdownRender>{str}</MarkdownRender>
                </div>);
            str = "\n";
        }
    };

    /* Function used to convert `elements` to container and add to `containers`. */
    const handleElements = () => {
        if (elements.length) {
            containers.push(createContainer(containerKey, elements));
            elements = [];
        }
    };

    mdSplit.map(strToElem).forEach((v) => {
        // console.log(v);
        if (typeof (v) !== "string") {

            handleStr();

            return elements.push(v);
        }

        let match = v.match(CONTAINER_RE);
        if (match) {
            handleStr();
            handleElements();

            containerKey = match[1] as CONTAINER_KEY;


        } else
            str += v + "\n";
    })

    // handle the string
    handleStr();

    // create last container
    handleElements();

    return containers;
}