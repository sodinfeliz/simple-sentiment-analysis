from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lib import calculate_naive_bayes, process_tweet, train_naive_bayes


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simple Sentiment Analysis")
        self.setFixedSize(600, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.text_input_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type your text here ...")
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.display_text)

        self.text_input_layout.addWidget(self.text_input)
        self.text_input_layout.addWidget(self.send_button)

        self.radio_buttons_layout = QHBoxLayout()
        self.naive_bayes_button = QRadioButton("Naive Bayes")
        self.naive_bayes_button.setChecked(True)
        self.logistic_regression_button = QRadioButton("Logistic Regression")

        self.radio_buttons_layout.addWidget(self.naive_bayes_button)
        self.radio_buttons_layout.addWidget(self.logistic_regression_button)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.naive_bayes_button)
        self.button_group.addButton(self.logistic_regression_button)

        self.display_window = QTextEdit()
        self.display_window.setReadOnly(True)

        self.layout.addLayout(self.text_input_layout)
        self.layout.addLayout(self.radio_buttons_layout)
        self.layout.addWidget(self.display_window)

    def display_text(self):
        text = self.text_input.text()
        processed_text = process_tweet(text)

        self.display_window.append(f"<b>Original</b>: {text}")
        self.display_window.append(f"<b>Processed</b>: {processed_text}")
        self.display_window.append("------------------------------------")

        if self.naive_bayes_button.isChecked():
            if not Path("pretrained/nb_model.json").exists():
                self.display_window.append("Training Naive Bayes model ...")
                train_naive_bayes()
                self.display_window.append("Training completed")
                self.display_window.append("------------------------------------")

            score = calculate_naive_bayes(text)
            sentiment = "😀" if score > 0 else ("😡" if score < 0 else "😶")

            self.display_window.append(f"<b>Score</b>: {score}")
            self.display_window.append(f"<b>Sentiment</b>: {sentiment}")
        elif self.logistic_regression_button.isChecked():
            ...

        self.display_window.append("====================================")
        self.text_input.clear()


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
