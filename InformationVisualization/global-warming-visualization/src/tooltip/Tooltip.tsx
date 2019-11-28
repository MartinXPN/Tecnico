import {Component} from "react";
import * as d3 from "d3";
import './Tooltip.css';

interface Props {
}
interface State {
}


export default class Tooltip extends Component<Props, State> {
    protected tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>;

    constructor(props: Props) {
        super(props);

        this.tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip");
        this.hide();
    }

    show = (html: string) => {
        this.tooltip.style("visibility", "visible");
        this.tooltip.html(html);
    };

    hide = () => {
        this.tooltip.style("visibility", "hidden");
    };

    move = (top: number, left: number) => {
        this.tooltip.transition()
            .duration(10)
            .style("top", top + "px")
            .style("left", left + "px");
    }
}
