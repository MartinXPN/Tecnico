import React, {Component} from "react";
import * as d3 from "d3";

interface Props {
    width: number | string;
    height: number | string;
    data: d3.DSVParsedArray<{ year: number, level: number, mass: number }>;
    yearStart: number;
    yearEnd: number;
}

function scaleRadial(domain: number[], range: number[]) {

    const scale = (x: number) => {
        const r0 = range[0] * range[0], r1 = range[1] * range[1];
        return Math.sqrt((x - domain[0]) / (domain[1] - domain[0]) * (r1 - r0) + r0);
    };

    scale.domain = (_: number[]) => _.length ? (domain = [+_[0], +_[1]], scale) : domain.slice();
    scale.range = (_: number[]) => _.length ? (range = [+_[0], +_[1]], scale) : range.slice();
    scale.ticks = (count: number) => d3.scaleLinear().domain(domain).ticks(count);
    scale.tickFormat = (count: number, specifier: any) => d3.scaleLinear().domain(domain).tickFormat(count, specifier);
    return scale;
}

export default class RadialBarChart extends Component<Props> {
    // @ts-ignore
    private ref: SVGSVGElement;
    private seaLevelElements: any;
    private glacierElements: any;

    addRadialChart = <T extends unknown>(elementSet: any,
                                         innerRadius: number, outerRadius: number,
                                         color: string, range: number[],
                                         data: d3.DSVParsedArray<T>,
                                         getValue: (d: any) => number, getX: (d: any) => number) => {

        const smallest = d3.min(data, getValue);
        const largest = d3.max(data, getValue);

        const x = d3.scaleBand()
            .range([range[0], range[1]])    // X axis goes from 0 to 2pi = all around the circle. If I stop at 1Pi, it will be around a half circle
            .domain(data.map(d => "" + getX(d))); // The domain of the X axis is the list of states.

        const y = scaleRadial(
            // @ts-ignore
            [smallest, largest],
            [innerRadius, outerRadius]
        );

        elementSet.attr("fill", color)
            .attr("d", d3.arc()     // imagine your doing a part of a donut plot
                .innerRadius(innerRadius)
                .outerRadius(d => {
                    // @ts-ignore
                    return (getX(d) < this.props.yearStart || getX(d) > this.props.yearEnd) ? y(smallest) : y(getValue(d));
                })
                // @ts-ignore
                .startAngle(d => {return x("" + getX(d));}).endAngle(d => {return x("" + getX(d)) + x.bandwidth();})
                .padAngle(0.01)
                .padRadius(innerRadius));
    };

    componentDidMount(): void {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        console.log(h, w);
        console.log(this.props.data);

        const svg = d3.select(this.ref)
            .append("g")
            .attr("transform", "translate(" + w / 2 + "," + h / 2 + ")"); // Add 100 on Y translation, cause upper bars are longer;

        this.seaLevelElements = svg.append("g")
            .selectAll("path")
            .data(this.props.data)
            .enter()
            .append("path");

        this.glacierElements = svg.append("g")
            .selectAll("path")
            .data(this.props.data)
            .enter()
            .append("path");
    }

    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        const innerRadius = Math.min(w, h) / 4;
        const outerRadius = Math.min(w, h) / 2;   // the outerRadius goes from the middle of the SVG area to the border

        this.addRadialChart(this.seaLevelElements, innerRadius, outerRadius, "#1484b3", [1.5 * Math.PI, 2.5 * Math.PI], this.props.data, (d) => d.level, d => d.year);
        this.addRadialChart(this.glacierElements, innerRadius, outerRadius, "#b32019", [-0.5 * Math.PI, -1.5 * Math.PI], this.props.data, (d) => -d.mass, d => d.year);
    }

    render(): React.ReactElement {
        return (
            <svg className="container"
                 ref={(ref: SVGSVGElement) => this.ref = ref}
                 width={this.props.width}
                 height={this.props.height}>
            </svg>
        );
    }
}
