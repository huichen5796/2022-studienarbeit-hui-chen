import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteTechnologyComponent } from './seite-technology.component';

describe('SeiteTechnologyComponent', () => {
  let component: SeiteTechnologyComponent;
  let fixture: ComponentFixture<SeiteTechnologyComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteTechnologyComponent]
    });
    fixture = TestBed.createComponent(SeiteTechnologyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
