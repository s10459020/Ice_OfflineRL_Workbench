from view_result import boxplot
from view_result import plot
from view_result import returns
from view_result import table


def main() -> None:
    returns.main()
    table.main()
    boxplot.main()
    plot.main()


if __name__ == "__main__":
    main()
